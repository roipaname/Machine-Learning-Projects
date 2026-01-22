from sqlalchemy import Column,Text,Integer,Float,DateTime,Boolean,Index,ForeignKey,Enum
from sqlachemy.ext.declarative import declarative_base
from slqalchemy.dialects.postgresql import JSON,UUID
import enum
from datetime import datetime
from  loguru import logger
import uuid


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



Base=declarative_base()


class Accounts(Base):
    __tablename__="account"
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
        AcquisitionChannel,"acquisition_channel_enum"
    ), nullable=True)
    customer_segment = Column(Enum(
        CustomerSegment,"customer_segmnet_enum"
    ), nullable=True)
    __table_args__=(
        Index("idx_customer_acquisition_channel","acquisition_channel"),
        Index("idx_customer_customer_segment","customer_segment"),
        Index("idx_customer_signup_date","signup_date")
    )


class Subscriptions(Base):
    __tablename__="subscriptions"
    subscription_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    account_id=Column(UUID(as_uuid=True),ForeignKey("accounts.account_id",ondelete="CASCADE"))
    plan_name=Column(Enum(SubPlanName,"subplan_name_enum"),nullable=False,default="basic")
    monthlyfee=Column(Float,nullable=False)
    start_date=Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date=Column(DateTime, nullable=False)
    status=Column(Enum(SubscriptionStatus,"sub_satus_enum"),nullable=False,default="active")
    __table_args__=(
        Index("idx_subscription_plan_name","plan_name"),
        Index("idx_subscription_status","status")
    )


class UsageEvents(Base):
    __tablename__="usage_events"
    event_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    customer_id=Column(UUID(as_uuid=True),ForeignKey("customers.customer_id",ondelete="CASCADE"))
    event_type=Column(Enum(
        EventType,"event_type_enum"
    ),nullable=False, default="apicall")
    device_type=Column(Text,nullable=True, default="phone")
    __table_args__=(
        Index("idx_usage_events_event_type","event_type"),
        Index("idx_usage_events_device_type","device_type")
    )

