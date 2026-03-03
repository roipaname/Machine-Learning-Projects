
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import InferenceClient
from src.ai_advisor.rag.document_store import RAGDocumentStore
from config.settings import HF_TOKEN,HF_MODEL
from src.ai_advisor.context_builder import CustomerContextBuilder
# ── Config ────────────────────────────────────────────────────────────────────
from loguru import logger

# ── Core advisor ──────────────────────────────────────────────────────────────
class ChurnAdvisor:
    def __init__(self, seed: bool = False):
        self.store  = RAGDocumentStore()
        self.client = InferenceClient(api_key=HF_TOKEN)
        self.context_builder = CustomerContextBuilder()

    def _build_customer_summary(self, context: dict) -> str:
        """
        Build a structured natural-language summary from CustomerContextBuilder output.
        """
        account   = context.get("account_info", {})
        engage    = context.get("engagement_signals", {})
        support   = context.get("support_signals", {})
        billing   = context.get("billing_signals", {})
        drivers   = context.get("top_churn_drivers", [])

        # Format top churn drivers as a readable list
        drivers_str = ", ".join(
            f"{d['feature']}={d['value']}" for d in drivers
        ) if drivers else "N/A"

        return (
            f"Customer ID:          {context.get('customer_id', 'Unknown')}\n"
            f"Churn Probability:    {context.get('churn_probability', 'N/A'):.0%}\n"
            f"Risk Tier:            {context.get('risk_tier', 'N/A')}\n"
            "\n── Account ──\n"
            f"  Tier:               {account.get('account_tier', 'N/A')}\n"
            f"  Segment:            {account.get('customer_segment', 'N/A')}\n"
            f"  Contract Type:      {account.get('contract_type', 'N/A')}\n"
            f"  Company Size:       {account.get('company_size', 'N/A')}\n"
            f"  Monthly Fee:        ${account.get('monthly_fee', 0)}\n"
            f"  Days to Renewal:    {account.get('days_until_renewal', 'N/A')}\n"
            "\n── Engagement ──\n"
            f"  Usage (30d):        {engage.get('usage_count_30d', 'N/A')}\n"
            f"  Usage (60d):        {engage.get('usage_count_60d', 'N/A')}\n"
            f"  Usage Decline:      {engage.get('usage_decline_30d_vs_60d', 'N/A')}\n"
            f"  Days Since Active:  {engage.get('days_since_last_activity', 'N/A')}\n"
            "\n── Support ──\n"
            f"  Tickets (30d):      {support.get('tickets_30d', 'N/A')}\n"
            f"  High-Pri Tickets:   {support.get('high_priority_tickets_90d', 'N/A')}\n"
            f"  Avg Satisfaction:   {support.get('avg_satisfaction_score', 'N/A')}\n"
            "\n── Billing ──\n"
            f"  Unpaid Invoices:    {billing.get('unpaid_invoices', 'N/A')}\n"
            f"  Avg Days Late:      {billing.get('avg_days_late', 'N/A')}\n"
            "\n── Top Churn Drivers ──\n"
            f"  {drivers_str}"
        )

    def advise(self, customer_id: str) -> str:
        """
        Main entry point. Pass a customer dict, get back advisory text.
        """
        
        context = self.context_builder.build_context(customer_id)
        # Guard: if context builder returned no data
        if context.get("context") == "No data available to build context":
            return f"⚠️  No data found for customer {customer_id}. Cannot generate advice."


        summary = self._build_customer_summary(context)

        # Pass structured context to RAG retrieval for smarter matching
        rag_context = {
            "account_info": context.get("account_info", {}),
            "risk_tier":    context.get("risk_tier", "unknown"),
        }

        docs = self.store.retrieve(query=summary, top_k=3, customer_context=rag_context)

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

        try:
            response = self.client.chat_completion(
            messages=messages,
            model=HF_MODEL,
            max_tokens=2000,
            temperature=0.3,)
            return response.choices[0].message.content.strip()
        except HfHubHTTPError as e:
            logger.error(f"LLM service unavailable: {e}")
            return "⚠️ AI advisor temporarily unavailable. Please retry."



# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set seed=True only the first time to populate ChromaDB
    advisor = ChurnAdvisor(seed=False)

    customer_id = "10eb03a7-efed-44d5-bb08-a23f9af1df08"  

    print("\n" + "=" * 60)
    print("CHURN ADVISOR REPORT")
    print("=" * 60)
    advice = advisor.advise(customer_id)
    print(advice)