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

    skill_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )

    skill: Mapped["Skill"] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("alias", name="uq_skill_aliases_alias"),
        Index("ix_skill_aliases_skill_id", "skill_id"),
    )

    def __repr__(self) -> str:
        return f"<SkillAlias alias={self.alias!r} → skill_id={self.skill_id}>"

# ---------------------------------------------------------------------------
# job_skills  (many-to-many: job_postings ↔ skills)
# ---------------------------------------------------------------------------

class JobSkill(Base):
    """
    Junction table linking job postings to extracted skills.
    Stores NLP extraction context for downstream analytics.
    """

    __tablename__ = "job_skills"

    job_skill_id:Mapped[UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)
    job_posting_id:Mapped[UUID]=mapped_column(
        UUID(as_uuid=True),ForeignKey("job_postings.job_posting_id",ondelete="CASCADE"),nullable=False
    )
    skill_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )
    # NLP extraction data
    frequency_mentioned: Mapped[int]           = mapped_column(Integer, default=1, nullable=False)
    context_snippet:     Mapped[Optional[str]] = mapped_column(Text)          # sentence(s) where skill appeared
    confidence_score:    Mapped[Optional[float]] = mapped_column(Float)       # NLP confidence 0.0–1.0
    is_required:         Mapped[Optional[bool]]  = mapped_column(Boolean)     # required vs. nice-to-have
    extraction_method:   Mapped[Optional[str]]   = mapped_column(String(64))  # "regex" | "ner" | "llm"

    # Relationships
    job_posting: Mapped["JobPosting"] = relationship(back_populates="job_skills")
    skill:       Mapped["Skill"]      = relationship(back_populates="job_skills")

    __table_args__ = (
        UniqueConstraint("job_posting_id", "skill_id", name="uq_job_skills_posting_skill"),
        Index("ix_job_skills_skill_id",       "skill_id"),
        Index("ix_job_skills_job_posting_id", "job_posting_id"),
        Index("ix_job_skills_confidence",     "confidence_score"),
    )

    def __repr__(self) -> str:
        return f"<JobSkill job={self.job_posting_id} skill={self.skill_id} freq={self.frequency_mentioned}>"
# ---------------------------------------------------------------------------
 #job_categories
# ---------------------------------------------------------------------------

class JobCategory(Base):
    """
    Categorical tags attached to a job posting.
    Covers seniority, domain (backend / ML / devops …), and remote type.
    """

    __tablename__ = "job_categories"

    job_cat_id: Mapped[UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    job_posting_id:Mapped[UUID]=mapped_column(
        UUID(as_uuid=True),ForeignKey("job_postings.job_posting_id",ondelete="CASCADE"),nullable=False
    )

    # Tag type + value stored as flexible string pairs
    tag_type:  Mapped[str] = mapped_column(String(64),  nullable=False)   # "domain" | "seniority" | "remote"
    tag_value: Mapped[str] = mapped_column(String(128), nullable=False)
    source:    Mapped[Optional[str]] = mapped_column(String(64))          # "nlp" | "manual" | "scraped"

    # Relationship
    job_posting: Mapped["JobPosting"] = relationship(back_populates="categories")

    __table_args__ = (
        UniqueConstraint("job_posting_id", "tag_type", "tag_value", name="uq_job_categories"),
        Index("ix_job_categories_tag_type",  "tag_type",  "tag_value"),
        Index("ix_job_categories_posting_id", "job_posting_id"),
    )

    def __repr__(self) -> str:
        return f"<JobCategory job={self.job_posting_id} {self.tag_type}={self.tag_value!r}>"


# ---------------------------------------------------------------------------
# skill_trends
# ---------------------------------------------------------------------------

class SkillTrend(Base):
    """
    Pre-aggregated time-series analytics per skill.
    Populated by the scheduler, NOT computed on-the-fly.
    """

    __tablename__ = "skill_trends"

    skill_trend_id: Mapped[UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    skill_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )

    period:       Mapped[TrendPeriod] = mapped_column(
                      Enum(TrendPeriod, name="trend_period_enum"), nullable=False
                  )
    period_start: Mapped[datetime]    = mapped_column(DateTime(timezone=True), nullable=False)
    period_end:   Mapped[datetime]    = mapped_column(DateTime(timezone=True), nullable=False)

    # Demand metrics
    job_count:     Mapped[int]            = mapped_column(Integer,  default=0, nullable=False)
    demand_score:  Mapped[Optional[float]] = mapped_column(Float)   # normalised 0–100
    growth_rate:   Mapped[Optional[float]] = mapped_column(Float)   # % change vs previous period

    # Compensation signals
    avg_salary_min: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    avg_salary_max: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))

    # Co-occurrence (top skills mentioned alongside this one)
    co_occurring_skills: Mapped[Optional[list]] = mapped_column(ARRAY(UUID))   # skill IDs

    # Top companies hiring for this skill this period
    top_companies: Mapped[Optional[list]] = mapped_column(ARRAY(UUID))         # company IDs

    # Geographic distribution  {"US": 400, "UK": 120, ...}
    geo_distribution: Mapped[Optional[dict]] = mapped_column(JSONB)

    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    skill: Mapped["Skill"] = relationship(back_populates="skill_trends")

    __table_args__ = (
        UniqueConstraint("skill_id", "period", "period_start", name="uq_skill_trends_skill_period"),
        Index("ix_skill_trends_period",      "skill_id", "period"),
        Index("ix_skill_trends_period_start", "period_start"),
        Index("ix_skill_trends_demand",       "demand_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<SkillTrend skill={self.skill_id} period={self.period} "
            f"start={self.period_start.date()} jobs={self.job_count}>"
        )


# ---------------------------------------------------------------------------
# scrape_runs
# ---------------------------------------------------------------------------

class ScrapeRun(Base):
    """Audit log for every scraper execution."""

    __tablename__ = "scrape_runs"

    id: Mapped[UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    source:  Mapped[JobSource]   = mapped_column(
                 Enum(JobSource, name="scrape_source_enum"), nullable=False
             )
    status:  Mapped[ScrapeStatus] = mapped_column(
                 Enum(ScrapeStatus, name="scrape_status_enum"),
                 default=ScrapeStatus.RUNNING,
                 nullable=False,
             )

    # Config snapshot used for this run
    config: Mapped[Optional[dict]] = mapped_column(JSONB)  # {"keywords": [...], "location": "..."}

    # Counters
    jobs_found:   Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_new:     Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Error handling
    error_log:    Mapped[Optional[str]] = mapped_column(Text)
    error_count:  Mapped[int]           = mapped_column(Integer, default=0, nullable=False)

    # Timing
    started_at:  Mapped[datetime]           = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    job_postings: Mapped[List["JobPosting"]] = relationship(back_populates="scrape_run")

    __table_args__ = (
        Index("ix_scrape_runs_source",     "source", "started_at"),
        Index("ix_scrape_runs_status",     "status"),
        Index("ix_scrape_runs_started_at", "started_at"),
    )

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def __repr__(self) -> str:
        return (
            f"<ScrapeRun id={self.id} source={self.source} "
            f"status={self.status} jobs_new={self.jobs_new}>"
        )


# ---------------------------------------------------------------------------
# Convenience: expose all models for Alembic autogenerate
# ---------------------------------------------------------------------------

__all__ = [
    "Base",
    # Enums
    "JobSource",
    "SeniorityLevel",
    "RemoteType",
    "TrendPeriod",
    "ScrapeStatus",
    "CompanySize",
    # Models
    "Company",
    "JobPosting",
    "SkillCategory",
    "Skill",
    "SkillAlias",
    "JobSkill",
    "JobCategory",
    "SkillTrend",
    "ScrapeRun",
]
