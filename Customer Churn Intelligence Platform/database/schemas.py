from sqlalchemy import Column,Text,Integer,Float,DateTime,Boolean,Index,ForeignKey,Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON,UUID
import enum
from datetime import datetime
from  loguru import logger
import uuid


# =========================
# ENUM DEFINITIONS
# =========================
# Enums enforce valid values at BOTH app and DB level
class ContractType(enum.Enum):
    monthly="monthly"
    annual="annual"

class AccountTier(enum.Enum):
    bronze="bronze"
    silver="silver"
    gold="gold"

class AcquisitionChannel(enum.Enum):
    ads="ads"
    referral="referral"
    sales="sales"
    organic="organic"

class CustomerSegment(enum.Enum):
    SMB="SMB"
    Midmarket="Midmarket"
    Enterprise="Enterprise"

class SubPlanName(enum.Enum):
    basic="basic"
    pro="pro"
    enterprise="enterprise"
class  SubscriptionStatus(enum.Enum):
    active="active"
    cancelled="cancelled"
    paused="paused"

class EventType(enum.Enum):
    login="login"
    apicall="apicall"
    upload="upload"
    export="export"
class Priority(enum.Enum):
    high="high"
    medium="medium"
    low="low"

class IssueType(enum.Enum):
    bug="bug"
    billing="billing"
    onboarding="onboarding"


Base=declarative_base()

# =========================
# ACCOUNTS TABLE
# =========================
# Represents a company / organization (B2B root entity)

class Accounts(Base):
    __tablename__="accounts"
    account_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    company_name=Column(Text,nullable=True)
    industry=Column(Text,nullable=True)
    company_size=Column(Integer,nullable=True)
    contract_type =Column(Enum(ContractType,name="contract_type_enum"),nullable=False)
    account_tier=Column(Enum(AccountTier,name="account_tier_enum"),nullable=False)
    created_at=Column(DateTime,nullable=False,default=datetime.utcnow)

    __table_args__=(
        Index("idx_accounts_contract_type","contract_type"),
        Index("idx_accounts_account_tier","account_tier"),
        Index("idx_accounts_industry","industry"),
        Index("idx_accounts-created_at","created_at")

    )
# =========================
# CUSTOMERS TABLE
# =========================
# Represents individual users belonging to an account
class Customers(Base):
    __tablename__="customers"
    customer_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    account_id=Column(UUID(as_uuid=True),ForeignKey("accounts.account_id",ondelete="CASCADE"))
    first_name = Column(Text, nullable=True)
    last_name = Column(Text, nullable=True)
    email = Column(Text, nullable=False, unique=True)
    country = Column(Text, nullable=True)

    signup_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    acquisition_channel = Column(Enum(
        AcquisitionChannel,name="acquisition_channel_enum"
    ), nullable=True)
    customer_segment = Column(Enum(
        CustomerSegment,name="customer_segmnet_enum"
    ), nullable=True)
    __table_args__=(
        Index("idx_customer_acquisition_channel","acquisition_channel"),
        Index("idx_customer_customer_segment","customer_segment"),
        Index("idx_customer_signup_date","signup_date")
    )

# =========================
# SUBSCRIPTIONS TABLE
# =========================
# Tracks billing plans at the account level
class Subscriptions(Base):
    __tablename__="subscriptions"
    subscription_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    account_id=Column(UUID(as_uuid=True),ForeignKey("accounts.account_id",ondelete="CASCADE"))
    plan_name=Column(Enum(SubPlanName,name="subplan_name_enum"),nullable=False,default="basic")
    monthlyfee=Column(Float,nullable=False)
    start_date=Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date=Column(DateTime, nullable=False)
    status=Column(Enum(SubscriptionStatus,name="sub_satus_enum"),nullable=False,default="active")
    __table_args__=(
        Index("idx_subscription_plan_name","plan_name"),
        Index("idx_subscription_status","status")
    )


# =========================
# USAGE EVENTS TABLE
# =========================
# High-volume behavioral data (key churn driver)
class UsageEvents(Base):
    __tablename__="usage_events"
    event_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    customer_id=Column(UUID(as_uuid=True),ForeignKey("customers.customer_id",ondelete="CASCADE"))
    event_type=Column(Enum(
        EventType,name="event_type_enum"
    ),nullable=False, default="apicall")
    device_type=Column(Text,nullable=True, default="phone")
    __table_args__=(
        Index("idx_usage_events_event_type","event_type"),
        Index("idx_usage_events_device_type","device_type")
    )
# =========================
# BILLING INVOICES TABLE
# =========================
# Payment behavior strongly predicts churn

class BillingInvoices(Base):
    __tablename__="billing_invoices"
    invoice_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    account_id=Column(UUID(as_uuid=True),ForeignKey("accounts.account_id",ondelete="CASCADE"))
    invoice_date=Column(DateTime,nullable=False,default=datetime.utcnow)
    paid=Column(Boolean,nullable=True)
    days_late=Column(Integer,nullable=True)

    __table_args__=(
        Index("idx_billing_invoices_invoice_date","invoice_date"),
        Index("idx_billing_invoices_days_late","days_late")
    )

class SupportTickets(Base):
    __tablename__="support_tickets"
    ticket_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    customer_id=Column(UUID(as_uuid=True),ForeignKey("customers.customer_id",ondelete="CASCADE"))
    created_at=Column(DateTime,nullable=False, default=datetime.utcnow)
    issue_type=Column(Enum(
        IssueType,name="issue_type_enum"
    ),nullable=False)
    priority=Column(Enum(
        Priority,name="priority_enum"
    ),nullable=False, default="low")
    resolution_time_hours=Column(Float,nullable=False)
    satisfaction_score=Column(Float,nullable=False)

    __table_args__=(
        Index("idx_support_ticket_issue_type","issue_type"),
        Index("idx_support_ticket_priority","priority")
    )


class Churn_Labels(Base):
    __tablename__="churn_labels"
    churn_id=Column(UUID(as_uuid=True),primary_key=True,unique=True, default=uuid.uuid4)
    customer_id=Column(UUID(as_uuid=True),ForeignKey("customers.customer_id",ondelete="CASCADE"),nullable=False)
    churned=Column(Boolean,default=False)
    churn_date=Column(DateTime, default=datetime.utcnow, nullable=False)
    churn_reason=Column(Text,nullable=True)

    __table_args_=(
        Index("idx_churn_labels_churn_outcome","churned")
    )