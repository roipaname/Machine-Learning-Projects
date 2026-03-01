from sqlalchemy import Column,Integer, Float,Text,DateTime,Boolean,Index,ForeignKey,Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postsegrl import JSON,UUID
import enum
from datetime import datetime
from loguru import logger
import uuid

Base=declarative_base()

# =========================
# ENUM DEFINITIONS
# =========================
# Enums enforce valid values at BOTH app and DB level
class Roles(enum.Enum):
    admin="admin"
    free_client="free_client"
class prefix(enum.Enum):
    mr="mr"
    mrs="mrs"
    dr="dr"
    ms="ms"
    honorable="honorable"


class Customer(Base):
    __tablename__="customers"
    customer_id=Column(UUID(as_uuid=True),primary_key=True,unique=True,default=uuid.uuid4)
    first_name=Column(Text,nullable=True)
    last_name=Column(Text,nullable=True)
    prefix=Column(Enum(
        prefix,name="prefix_enum"
    ),nullable=True)
    sexe=Column(Text)
    role=Column(Enum(
        Roles,name="roles_enum"
    ),nullable=False,default="free_client")
    created_at=Column(DateTime,default=datetime.utcnow,nullable=False)
    __table_args__=(
        Index("idx_customers_role","role"),
        Index("idx_customers_prefix","prefix"),
        Index("idx_customers_fname","first_name"),
        Index("idx_customers_lname","last_name")
    )
