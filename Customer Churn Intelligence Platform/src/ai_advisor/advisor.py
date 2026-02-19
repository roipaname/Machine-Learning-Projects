

from huggingface_hub import InferenceClient
from src.ai_advisor.rag.document_store import RAGDocumentStore
from config.settings import HF_TOKEN,HF_MODEL
from src.ai_advisor.context_builder import CustomerContextBuilder
# ── Config ────────────────────────────────────────────────────────────────────


# ── Core advisor ──────────────────────────────────────────────────────────────
class ChurnAdvisor:
    def __init__(self, seed: bool = False):
        self.store  = RAGDocumentStore()
        self.client = InferenceClient(api_key=HF_TOKEN)

    def _build_customer_summary(self, customer: dict) -> str:
        return (
            f"Customer: {customer.get('name', 'Unknown')}\n"
            f"Segment / Tier: {customer.get('tier', 'N/A')}\n"
            f"MRR: ${customer.get('mrr', 0)}\n"
            f"Login trend: {customer.get('login_trend', 'N/A')}\n"
            f"Support tickets: {customer.get('tickets', 'N/A')}\n"
            f"Feature usage: {customer.get('feature_usage', 'N/A')}\n"
            f"Days to renewal: {customer.get('days_to_renewal', 'N/A')}\n"
            f"Last sentiment: {customer.get('last_sentiment', 'N/A')}"
        )

    def advise(self, customer_id: str) -> str:
        """
        Main entry point. Pass a customer dict, get back advisory text.
        """
        customer_context_builder=CustomerContextBuilder()
        customer_data=customer_context_builder.build_context(customer_id=customer_id)
        summary = self._build_customer_summary(customer_data)

        context = {
            "account_info": {"tier": customer_data.get("tier")},
            "risk_tier": customer_data.get("risk_tier", "unknown"),
        }

        docs = self.store.retrieve(query=summary, top_k=3, customer_context=context)

        context_block = "\n\n".join(
            f"[{doc['doc_id']}] (relevance score: {doc['score']:.3f})\n{doc['content']}"
            for doc in docs
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert customer success churn advisor. "
                    "Use the provided strategy documents to give specific, actionable advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Relevant strategy documents:\n{context_block}\n\n"
                    f"## Current customer situation:\n{summary}\n\n"
                    "## Your task:\n"
                    "1. Churn Risk: [Low / Medium / High / Critical]\n"
                    "2. Top 3 risk signals\n"
                    "3. Recommended intervention (cite the relevant playbook/strategy)\n"
                    "4. Act within: X days\n"
                    "5. Predicted outcome if no action taken\n\n"
                    "Be concise, specific, and actionable."
                ),
            },
        ]

        response = self.client.chat_completion(
            messages=messages,
            model=HF_MODEL,
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set seed=True only the first time to populate ChromaDB
    advisor = ChurnAdvisor(seed=True)

    customer = {
        "name":            "Acme Corp",
        "tier":            "Enterprise",
        "mrr":             8500,
        "login_trend":     "down 65% over last 30 days",
        "tickets":         "3 billing complaints this month",
        "feature_usage":   "stopped using core dashboard feature",
        "days_to_renewal": 38,
        "last_sentiment":  "frustrated, mentioned switching to competitor",
        "risk_tier":       "High",
    }

    print("\n" + "="*60)
    print("CHURN ADVISOR REPORT")
    print("="*60)
    advice = advisor.advise(customer)
    print(advice)