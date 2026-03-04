from sqlalchemy import (
    Column,Integer, Float,Text,DateTime,Boolean,Index,ForeignKey,Enum,
    Numeric,String,UniqueConstraint,func,BigInteger
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON,UUID,ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum
from datetime import datetime
from loguru import logger
import uuid
from typing import List,Optional
Base=declarative_base()

# =========================
# ENUM DEFINITIONS
# =========================
# Enums enforce valid values at BOTH app and DB level

class JobSource(str, enum.Enum):
    LINKEDIN = "linkedin"
    INDEED   = "indeed"
    GLASSDOOR = "glassdoor"
    REMOTEOK  = "remoteok"
    OTHER     = "other"
class SeniorityLevel(str,enum.Enum):
    INTERN     = "intern"
    JUNIOR     = "junior"
    MID        = "mid"
    SENIOR     = "senior"
    LEAD       = "lead"
    PRINCIPAL  = "principal"
    MANAGER    = "manager"
    DIRECTOR   = "director"
    VP         = "vp"
    EXECUTIVE  = "executive"
    UNKNOWN    = "unknown"

class RemoteType(str, enum.Enum):
    ON_SITE = "on_site"
    HYBRID  = "hybrid"
    REMOTE  = "remote"
    UNKNOWN = "unknown"


class TrendPeriod(str, enum.Enum):
    WEEKLY  = "weekly"
    MONTHLY = "monthly"

class ScrapeStatus(str, enum.Enum):
    RUNNING   = "running"
    SUCCESS   = "success"
    PARTIAL   = "partial"
    FAILED    = "failed"


class CompanySize(str, enum.Enum):
    STARTUP     = "1-10"
    SMALL       = "11-50"
    MEDIUM      = "51-200"
    LARGE       = "201-1000"
    ENTERPRISE  = "1001-5000"
    MEGA        = "5000+"
    UNKNOWN     = "unknown"

class Company(Base):
    __tablename__="company"
    company_id=Column(UUID(as_uuid=True),primary_key=True,nullable=False,default=uuid.uuid4)
    company_name=Column(Text,nullable=False)
    normalized_name=Column(Text,nullable=False)
    industry=Column(Text,nullable=True)
    company_size=Column(Enum(
        CompanySize,name="company_size_enum"
    ),nullable=False)
    hq_location=Column(Text,nullable=True)
    linkedin_url=Column(Text,nullable=True)
    created_at=Column(DateTime,default=datetime.utcnow,nullable=False)
    updated_at=Column(DateTime,default=datetime.utcnow,nullable=False)
    job_postings: Mapped[List["JobPosting"]] = relationship(
        "JobPosting",
        back_populates="company",
        cascade="all, delete-orphan"
    )

class JobPosting(Base):
#Attempting new style
    __tablename__="job_postings"
    posting_id=Column(UUID(as_uuid=True),primary_key=True,nullable=False,default=uuid.uuid4)
     # Content
    title:       Mapped[str]           = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    url:         Mapped[str]           = mapped_column(String(1024), nullable=False)

    # Location & remote
    remote_type: Mapped[RemoteType]    = mapped_column(
                     Enum(RemoteType, name="remote_type_enum"),
                     default=RemoteType.UNKNOWN,
                     nullable=False,
                 )

    #Compensations
    salary_min: Mapped[Optional[float]]=mapped_column(Numeric(12,2))
    salary_max: Mapped[Optional[float]]=mapped_column(Numeric(12,2))

    seniority:Mapped[SeniorityLevel]=mapped_column(Enum(SeniorityLevel,name="seniority_level_enum"),default=SeniorityLevel.UNKNOWN,nullable=False)
    trend_period:Mapped[Optional[TrendPeriod]]=mapped_column(Enum(TrendPeriod,name="Trend_period_enum"),nullable=True)
