"""
churn_advisor.py
----------------
Simplified churn advisor built on top of RAGDocumentStore.
Uses HuggingFace Inference API (free tier) for LLM reasoning.

Usage:
    python churn_advisor.py
"""

from huggingface_hub import InferenceClient
from src.ai_advisor.rag.document_store import RAGDocumentStore
from config.settings import HF_TOKEN

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ── Seed data (run once, then comment out) ────────────────────────────────────
SEED_DOCS = [
    {
        "doc_id": "strategy_001",
        "content": (
            "Q4 2024 Retention Analysis - Enterprise Tech Segment. "
            "For enterprise customers (MRR >$5k) showing engagement decline >40%: "
            "Executive Business Review within 5 days gives 73% save rate. "
            "Combined with product training it rises to 81%. "
            "Strategic discount 15-20% gives 68% save rate standalone. "
            "Best approach: EBR + training + conditional discount."
        ),
        "metadata": {"type": "analyst_report", "segment": "Enterprise", "success_rate": 0.73},
    },
    {
        "doc_id": "playbook_001",
        "content": (
            "High-Touch Intervention Playbook for customers with high support volume. "
            "Steps: acknowledge pain points, assign dedicated CSM for 90 days, "
            "weekly check-ins, product optimisation session within 2 weeks, "
            "executive sponsor call if no improvement after 30 days. "
            "Success rate: 65% for customers with >10 tickets/month."
        ),
        "metadata": {"type": "playbook", "focus": "support_intensive", "success_rate": 0.65},
    },
    {
        "doc_id": "playbook_002",
        "content": (
            "SMB Churn Prevention Playbook. "
            "For SMB customers (MRR <$1k) with login drop >50%: "
            "Send personalised email sequence (3 emails over 2 weeks). "
            "Offer free onboarding refresh webinar. "
            "If no response, offer 10% loyalty discount. "
            "Success rate: 48% for SMB segment."
        ),
        "metadata": {"type": "playbook", "segment": "SMB", "success_rate": 0.48},
    },
]


# ── Core advisor ──────────────────────────────────────────────────────────────
class ChurnAdvisor:
    def __init__(self, seed: bool = False):
        self.store  = RAGDocumentStore()
        self.client = InferenceClient(api_key=HF_TOKEN)

        if seed:
            self.store.add_documents(SEED_DOCS)
            print("✅ Seed documents loaded into ChromaDB")

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

    def advise(self, customer: dict) -> str:
        """
        Main entry point. Pass a customer dict, get back advisory text.
        """
        summary = self._build_customer_summary(customer)

        context = {
            "account_info": {"tier": customer.get("tier")},
            "risk_tier": customer.get("risk_tier", "unknown"),
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
            model=MODEL,
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