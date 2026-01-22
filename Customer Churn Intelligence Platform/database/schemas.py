from sqlalchemy import Column,Text,Integer,Float,DateTime,Boolean,Index,UniqueConstraint,Enum
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
    account_id=Column(UUID(as_uuid=True))
