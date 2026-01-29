import pandas as pd
import numpy as np
import uuid
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

# =========================
# LOAD RAW DATA
# =========================
telco = pd.read_csv("data/raw/telco_churn.csv")
usage = pd.read_csv("data/raw/usage_dataset.csv")
tickets = pd.read_csv("data/raw/customer_support_ticket.csv")
invoices = pd.read_csv("data/raw/invoices.csv")

# =========================
# 1️⃣ ACCOUNTS (1,000)
# =========================
N_ACCOUNTS = 1000

accounts = pd.DataFrame({
    "account_id": [uuid.uuid4() for _ in range(N_ACCOUNTS)],
    "company_name": [fake.company() for _ in range(N_ACCOUNTS)],
    "industry": np.random.choice(
        ["SaaS", "FinTech", "Health", "Retail","Entertainment","Tech"], N_ACCOUNTS
    ),
    "company_size": np.random.randint(10, 3000, N_ACCOUNTS),
    "contract_type": np.random.choice(["monthly", "annual"], N_ACCOUNTS),
    "account_tier": None,
    "created_at": pd.to_datetime(
        np.random.choice(
            pd.date_range("2017-01-01", "2024-01-01"), N_ACCOUNTS
        )
    )
})

accounts["account_tier"] = accounts["company_size"].apply(
    lambda x: "gold" if x > 1000 else "silver" if x > 200 else "bronze"
)

# =========================
# 2️⃣ CUSTOMERS (7,000)
# =========================
N_CUSTOMERS = 7000
telco_sample = telco.sample(N_CUSTOMERS, replace=True)

customers = pd.DataFrame({
    "customer_id": [uuid.uuid4() for _ in range(N_CUSTOMERS)],
    "account_id": np.random.choice(accounts["account_id"], N_CUSTOMERS),
    "first_name": [fake.first_name() for _ in range(N_CUSTOMERS)],
    "last_name": [fake.last_name() for _ in range(N_CUSTOMERS)],
    "email": [fake.unique.email() for _ in range(N_CUSTOMERS)],
    "country": np.random.choice(["US", "UK", "DE", "ZA"], N_CUSTOMERS),
    "signup_date": pd.to_datetime("2024-01-01") -
        pd.to_timedelta(telco_sample["tenure"].astype(int) * 30, unit="D"),
    "acquisition_channel": np.random.choice(
        ["ads", "referral", "sales", "organic"], N_CUSTOMERS
    ),
    "customer_segment": None
})

# segment from account size
acc_size_map = dict(zip(accounts["account_id"], accounts["company_size"]))
customers["customer_segment"] = customers["account_id"].map(
    lambda x: "Enterprise" if acc_size_map[x] > 1000
    else "Midmarket" if acc_size_map[x] > 200
    else "SMB"
)

# =========================
# 3️⃣ SUBSCRIPTIONS (1 per account)
# =========================
subscriptions = accounts[["account_id"]].copy()

subscriptions["subscription_id"] = [uuid.uuid4() for _ in range(N_ACCOUNTS)]
subscriptions["plan_name"] = np.random.choice(
    ["basic", "pro", "enterprise"], N_ACCOUNTS, p=[0.5, 0.3, 0.2]
)
subscriptions["monthlyfee"] = np.random.uniform(20, 200, N_ACCOUNTS).round(2)
subscriptions["start_date"] = accounts["created_at"]
subscriptions["end_date"] = subscriptions["start_date"] + pd.to_timedelta(365, unit="D")
subscriptions["status"] = np.random.choice(
    ["active", "cancelled", "paused"], N_ACCOUNTS, p=[0.7, 0.2, 0.1]
)

# =========================
# 4️⃣ USAGE EVENTS (15,000)
# =========================
N_EVENTS = 15000
usage_events = pd.DataFrame({
    "event_id": [uuid.uuid4() for _ in range(N_EVENTS)],
    "customer_id": np.random.choice(customers["customer_id"], N_EVENTS),
    "event_type": np.random.choice(
        ["login", "apicall", "upload", "export"], N_EVENTS
    ),
    "device_type": np.random.choice(
        usage["Operating System"], N_EVENTS
    ),
    "timestamp":pd.to_datetime(
        np.random.choice(
            pd.date_range("2017-01-01", "2024-01-01"), N_EVENTS
        )
    )
})

# =========================
# 5️⃣ SUPPORT TICKETS (4,000)
# =========================
N_TICKETS = 4000
ticket_sample = tickets.sample(N_TICKETS, replace=True)

support_tickets = pd.DataFrame({
    "ticket_id": [uuid.uuid4() for _ in range(N_TICKETS)],
    "customer_id": np.random.choice(customers["customer_id"], N_TICKETS),
    "created_at": pd.to_datetime(
        ticket_sample["Date of Purchase"], errors="coerce"
    ).fillna(pd.Timestamp.utcnow()),
    "issue_type": np.random.choice(
        ["bug", "billing", "onboarding"], N_TICKETS
    ),
    "priority": np.random.choice(
        ["low", "medium", "high"], N_TICKETS, p=[0.5, 0.3, 0.2]
    ),
    "resolution_time_hours": np.random.uniform(1, 72, N_TICKETS).round(2),
    "satisfaction_score": np.random.uniform(1, 5, N_TICKETS).round(2)
})

# =========================
# 6️⃣ BILLING INVOICES (2,000)
# =========================
N_INVOICES = 2000
invoice_sample = invoices.sample(N_INVOICES, replace=True)

billing_invoices = pd.DataFrame({
    "invoice_id": [uuid.uuid4() for _ in range(N_INVOICES)],
    "account_id": np.random.choice(accounts["account_id"], N_INVOICES),
    "invoice_date": pd.to_datetime(
        invoice_sample["invoice_date"], errors="coerce"
    ).fillna(pd.Timestamp.utcnow()),
    "paid": np.random.choice([True, False], N_INVOICES, p=[0.85, 0.15]),
    "days_late": np.random.poisson(7, N_INVOICES)
})

# =========================
# WRITE OUTPUT
# =========================
accounts.to_csv("data/processed/csv/accounts.csv", index=False)
customers.to_csv("data/processed/csv/customers.csv", index=False)
subscriptions.to_csv("data/processed/csv/subscriptions.csv", index=False)
usage_events.to_csv("data/processed/csv/usage_events.csv", index=False)
support_tickets.to_csv("data/processed/csv/support_tickets.csv", index=False)
billing_invoices.to_csv("data/processed/csv/billing_invoices.csv", index=False)

print("✅ CSVs created (≤ 30k rows total)")
