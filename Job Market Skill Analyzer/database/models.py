from sqlalchemy import (
    Column,Integer, Float,Text,DateTime,Boolean,Index,ForeignKey,Enum,
    Numeric,String,UniqueConstraint,func,BigInteger
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON,UUID,ARRAY,JSONB
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
    """Normalized employer / company data."""

    __tablename__ = "companies"

    company_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name:             Mapped[str]           = mapped_column(String(255), nullable=False)
    normalized_name:  Mapped[str]           = mapped_column(String(255), nullable=False, index=True)
    website:          Mapped[Optional[str]] = mapped_column(String(512))
    industry:         Mapped[Optional[str]] = mapped_column(String(128))
    size:             Mapped[CompanySize]   = mapped_column(
                          Enum(CompanySize, name="company_size_enum"),
                          default=CompanySize.UNKNOWN,
                          nullable=False,
                      )
    hq_location:      Mapped[Optional[str]] = mapped_column(String(255))
    linkedin_url:     Mapped[Optional[str]] = mapped_column(String(512))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    job_postings: Mapped[List["JobPosting"]] = relationship(back_populates="company")

    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_companies_normalized_name"),
    )

    def __repr__(self) -> str:
        return f"<Company id={self.company_id} name={self.name!r}>"

# ---------------------------------------------------------------------------
# job_postings
# ---------------------------------------------------------------------------

class JobPosting(Base):
    """Core scraped job entity."""

    __tablename__ = "job_postings"

    job_posting_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Content
    title:       Mapped[str]           = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    url:         Mapped[str]           = mapped_column(String(1024), nullable=False)

    # Location & remote
    location:    Mapped[Optional[str]] = mapped_column(String(255))
    remote_type: Mapped[RemoteType]    = mapped_column(
                     Enum(RemoteType, name="remote_type_enum"),
                     default=RemoteType.UNKNOWN,
                     nullable=False,
                 )

    # Compensation
    salary_min:      Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    salary_max:      Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    salary_currency: Mapped[Optional[str]]   = mapped_column(String(8))

    # Seniority (NLP-derived or parsed)
    seniority: Mapped[SeniorityLevel] = mapped_column(
                   Enum(SeniorityLevel, name="seniority_level_enum"),
                   default=SeniorityLevel.UNKNOWN,
                   nullable=False,
               )

    # Source & scraping metadata
    source:     Mapped[JobSource] = mapped_column(
                    Enum(JobSource, name="job_source_enum"),
                    nullable=False,
                )
    external_id: Mapped[Optional[str]] = mapped_column(String(255))   # ID on source platform
    scrape_run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("scrape_runs.id", ondelete="SET NULL"), nullable=True
    )

    # Flags
    is_active:   Mapped[bool] = mapped_column(Boolean, default=True,  nullable=False)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # NLP done?

    # Timestamps
    posted_at:  Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    scraped_at: Mapped[datetime]            = mapped_column(
                    DateTime(timezone=True), server_default=func.now()
                )
    updated_at: Mapped[datetime]            = mapped_column(
                    DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
                )

    # FK
    company_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.company_id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    company:     Mapped[Optional["Company"]]       = relationship(back_populates="job_postings")
    job_skills:  Mapped[List["JobSkill"]]           = relationship(back_populates="job_posting", cascade="all, delete-orphan")
    categories:  Mapped[List["JobCategory"]]        = relationship(back_populates="job_posting", cascade="all, delete-orphan")
    scrape_run:  Mapped[Optional["ScrapeRun"]]      = relationship(back_populates="job_postings")

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_job_postings_source_external"),
        Index("ix_job_postings_posted_at",  "posted_at"),
        Index("ix_job_postings_scraped_at", "scraped_at"),
        Index("ix_job_postings_source",     "source", "scraped_at"),
        Index("ix_job_postings_company",    "company_id"),
        Index("ix_job_postings_seniority",  "seniority"),
    )

    def __repr__(self) -> str:
        return f"<JobPosting id={self.job_posting_id} title={self.title!r} source={self.source}>"
# ---------------------------------------------------------------------------
# skill_categories
# ---------------------------------------------------------------------------

class SkillCategory(Base):
    """Taxonomy node for grouping skills (e.g. Programming, Cloud, Soft Skills)."""

    __tablename__ = "skill_categories"
    skill_category_id:Mapped[UUID]=mapped_column(UUID(as_uuid=True),default=uuid.uuid4,primary_key=True)
    name:Mapped[str]=mapped_column(String(128),nullable=False,unique=True)
    slug:        Mapped[str]           = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    parent_id:Mapped[Optional[UUID]]=mapped_column(
        UUID(as_uuid=True),ForeignKey("skill_categories.skill_category_id", ondelete="SET NULL"), nullable=True
    )

    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Self-referential relationship for nested categories
    parent:   Mapped[Optional["SkillCategory"]]  = relationship("SkillCategory", remote_side="SkillCategory.id", back_populates="children")
    children: Mapped[List["SkillCategory"]]       = relationship("SkillCategory", back_populates="parent")
    skills:   Mapped[List["Skill"]]               = relationship(back_populates="category")

    def __repr__(self) -> str:
        return f"<SkillCategory id={self.skill_category_id} name={self.name!r}>"
    
# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

class Skill(Base):
    """Master skill registry — canonical source of truth. """

    __tablename__ = "skills"
    
    skill_id:        Mapped[UUID]=mapped_column(UUID(as_uuid=True),default=uuid.uuid4,primary_key=True)
    name:            Mapped[str]           = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str]           = mapped_column(String(255), nullable=False)
    slug:            Mapped[str]           = mapped_column(String(255), nullable=False, unique=True)
    description:     Mapped[Optional[str]] = mapped_column(Text)

    # Taxonomy
    category_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("skill_categories.skill_category_id", ondelete="SET NULL"), nullable=True
    )

    # Quality flags
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active:   Mapped[bool] = mapped_column(Boolean, default=True,  nullable=False)

    # Optional structured metadata (e.g. {"type": "language", "paradigm": "OOP"})
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    category:      Mapped[Optional["SkillCategory"]] = relationship(back_populates="skills")
    aliases:       Mapped[List["SkillAlias"]]         = relationship(back_populates="skill", cascade="all, delete-orphan")
    job_skills:    Mapped[List["JobSkill"]]           = relationship(back_populates="skill")
    skill_trends:  Mapped[List["SkillTrend"]]         = relationship(back_populates="skill", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_skills_normalized_name"),
        Index("ix_skills_category", "category_id"),
    )

    def __repr__(self) -> str:
        return f"<Skill id={self.skill_id} name={self.name!r}>"




# ---------------------------------------------------------------------------
# skill_aliases
# ---------------------------------------------------------------------------


class SkillAlias(Base):
    """
    Alternate names / abbreviations for a skill.
    e.g.  JS → JavaScript,  ML → Machine Learning,  k8s → Kubernetes
    """

    __tablename__ = "skill_aliases"
    skill_alias_id:Mapped[UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    alias: Mapped[str] = mapped_column(String(255), nullable=False)

    skill_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )

    skill: Mapped["Skill"] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("alias", name="uq_skill_aliases_alias"),
        Index("ix_skill_aliases_skill_id", "skill_id"),
    )

    def __repr__(self) -> str:
        return f"<SkillAlias alias={self.alias!r} → skill_id={self.skill_id}>"