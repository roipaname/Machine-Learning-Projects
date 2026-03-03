#backend.main
from fastapi import FastAPI,HTTPException,BackgroundTasks,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional,Dict,Any,List
from datetime import datetime
from loguru import logger
import uuid
import shutil
import tempfile
from pathlib import Path
from src.utils.file_readers import read_file_to_text
from src.ai_advisor.advisor import ChurnAdvisor
from src.ai_advisor.context_builder import CustomerContextBuilder
from src.ai_advisor.rag.document_store import RAGDocumentStore
from config.settings import LOGS_DIR
from backend.helpers import _make_request_id,_now
from backend.schema import AdviceRequest,AdviceResponse,ContextResponse,BulkAdviceRequest,BulkAdviceResponse,HealthResponse,CustomerPredictionRequest
from database.operations import get_accounts_by_company_name
app=FastAPI(
    title="Customer Churn Intelligence Platform",
    description="AI-powered churn prediction and intervention advisor",
    version="1.0.0"
)

# CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

advisor:Optional[ChurnAdvisor]=None
context_builder:Optional[CustomerContextBuilder]=None
store:Optional[RAGDocumentStore]=None

@app.on_event("startup")
async def startup_event():
    global advisor,context_builder,store
    logger.info("Starting up application...")
    advisor=ChurnAdvisor()
    context_builder=CustomerContextBuilder()
    store=RAGDocumentStore()
    logger.success("Application startup complete")



def _check_ready():
    if advisor is None or context_builder is None:
        raise HTTPException(status_code=503, detail="Service not ready. Models still loading.")
    

def _check_store():
    if store is None:
        raise HTTPException(status_code=503, detail="RAG store not ready.")   
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        timestamp=_now(),
        version="1.0.0",
        models_loaded=advisor is not None and context_builder is not None,
    )

@app.get("/context/{customer_id}",response_model=ContextResponse,tags=["Context"])
async def get_customer_context(customer_id:str):
    """
    Return the ML-enriched customer context (churn probability, risk tier,
    signals, top drivers) without calling the LLM advisor.
    Fast and cheap — use this for dashboards or pre-screening.
    """
    _check_ready()

    context=context_builder.build_context(customer_id)
    if context.get("context") == "No data available to build context":
        raise HTTPException(status_code=404, detail=f"No data found for customer {customer_id}")
    return ContextResponse(
        request_id=_make_request_id(),
        customer_id=customer_id,
        timestamp=_now(),
        churn_probability=context["churn_probability"],
        risk_tier=context["risk_tier"],
        account_info=context.get("account_info", {}),
        engagement_signals=context.get("engagement_signals", {}),
        support_signals=context.get("support_signals", {}),
        billing_signals=context.get("billing_signals", {}),
        top_churn_drivers=context.get("top_churn_drivers", []),
    )

@app.get("/advice/{customer_id}",response_model=AdviceResponse,tags=["Advice"])
async def get_advice(customer_id:str):
    """
    Run the full churn advisor pipeline for a single customer:
    context build → RAG retrieval → LLM recommendation.
    """
    _check_ready()
    
    logger.info(f"Advice requested for customer {customer_id}")
    context=context_builder.build_context(customer_id)
    if context.get("context") == "No data available to build context":
        raise HTTPException(status_code=404, detail=f"No data found for customer {customer_id}")
    advice=advisor.advise(customer_id)
    return AdviceResponse(
        request_id=_make_request_id(),
        customer_id=customer_id,
        timestamp=_now(),
        churn_probability=context["churn_probability"],
        risk_tier=context["risk_tier"],
        advice=advice,
        account_info=context.get("account_info", {}),
        engagement_signals=context.get("engagement_signals", {}),
        support_signals=context.get("support_signals", {}),
        billing_signals=context.get("billing_signals", {}),
        top_churn_drivers=context.get("top_churn_drivers", []),
    )


@app.post("/advise/bulk", response_model=BulkAdviceResponse, tags=["Advisor"])
async def get_bulk_advice(request: BulkAdviceRequest):
    """
    Run the advisor for multiple customers (max 20 per request).
    Errors per customer are captured and returned without failing the whole batch.
    """
    _check_ready()

    results = []
    errors  = []

    for customer_id in request.customer_ids:
        try:
            context = context_builder.build_context(customer_id)

            if context.get("context") == "No data available to build context":
                errors.append({"customer_id": customer_id, "error": "No data found"})
                continue

            advice = advisor.advise(customer_id)

            results.append({
                "customer_id":       customer_id,
                "churn_probability": context["churn_probability"],
                "risk_tier":         context["risk_tier"],
                "advice":            advice,
                "account_info":      context.get("account_info", {}),
                "top_churn_drivers": context.get("top_churn_drivers", []),
            })

        except Exception as e:
            logger.error(f"Error processing customer {customer_id}: {e}")
            errors.append({"customer_id": customer_id, "error": str(e)})

    return BulkAdviceResponse(
        request_id=_make_request_id(),
        timestamp=_now(),
        total=len(results),
        results=results,
        errors=errors,
    )


@app.get("/advise/risk/{risk_tier}", tags=["Advisor"])
async def get_advice_by_risk_tier(
    risk_tier: str,
    customer_ids: str,  # comma-separated list passed as query param
):
    """
    Filter and advise only customers matching a given risk tier.
    Example: /advise/risk/Critical?customer_ids=id1,id2,id3
    """
    _check_ready()

    valid_tiers = {"Low", "Medium", "High", "Critical"}
    if risk_tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk tier. Must be one of: {valid_tiers}"
        )

    ids     = [cid.strip() for cid in customer_ids.split(",") if cid.strip()]
    matched = []
    errors  = []

    for customer_id in ids:
        try:
            context = context_builder.build_context(customer_id)
            if context.get("risk_tier") == risk_tier:
                advice = advisor.advise(customer_id)
                matched.append({
                    "customer_id":       customer_id,
                    "churn_probability": context["churn_probability"],
                    "risk_tier":         context["risk_tier"],
                    "advice":            advice,
                    "top_churn_drivers": context.get("top_churn_drivers", []),
                })
        except Exception as e:
            logger.error(f"Error processing customer {customer_id}: {e}")
            errors.append({"customer_id": customer_id, "error": str(e)})

    return {
        "request_id": _make_request_id(),
        "timestamp":  _now(),
        "risk_tier":  risk_tier,
        "matched":    len(matched),
        "results":    matched,
        "errors":     errors,
    }

@app.get("/accounts/{company_name}", tags=["Data"] )
async def get_accounts_by_company(company_name: str):
    """
    Example endpoint to demonstrate fetching accounts by company name.
    This can be used for testing database connectivity and query functionality.
    """
    try:
        accounts = get_accounts_by_company_name(company_name)
        return {"company_name": company_name, "accounts": [acc.__dict__ for acc in accounts]}
    except Exception as e:
        logger.error(f"Error fetching accounts for company {company_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Chunking helper ───────────────────────────────────────────

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Knowledge Base ────────────────────────────────────────────

@app.post("/knowledge-base/upload", tags=["Knowledge Base"])
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("other"),
    segment: str = Form("All"),
) -> dict:
    """
    Upload a .txt / .md / .pdf / .docx file, chunk it, and ingest
    all chunks into the RAG vector store.
    """
    _check_store()

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    # Persist upload to a temp file so file_readers can use a real Path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        text = read_file_to_text(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    chunks  = _chunk_text(text)
    base_id = f"upload_{uuid.uuid4().hex[:8]}"

    docs = [
        {
            "doc_id":   f"{base_id}_chunk{i}",
            "content":  chunk,
            "metadata": {
                "source":        file.filename,
                "doc_type":      doc_type,
                "segment":       segment,
                "chunk_index":   i,
                "total_chunks":  len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        store.add_documents(docs)
    except Exception as exc:
        logger.error(f"RAG ingestion failed for '{file.filename}': {exc}")
        raise HTTPException(status_code=500, detail="Failed to ingest document.")

    logger.success(f"Ingested {len(docs)} chunks from '{file.filename}' (base_id: {base_id})")
    return {
        "doc_id":       base_id,
        "filename":     file.filename,
        "chunks_added": len(docs),
        "doc_type":     doc_type,
        "segment":      segment,
    }