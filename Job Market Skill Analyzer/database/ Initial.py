"""Initial schema — UUID version matching SQLAlchemy models"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():

    # Enable UUID generation
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')

    # ---------------- ENUMS ----------------

    job_source_enum = postgresql.ENUM(
        "linkedin","indeed","glassdoor","remoteok","other",
        name="job_source_enum"
    )

    seniority_enum = postgresql.ENUM(
        "intern","junior","mid","senior","lead","principal",
        "manager","director","vp","executive","unknown",
        name="seniority_level_enum"
    )

    remote_enum = postgresql.ENUM(
        "on_site","hybrid","remote","unknown",
        name="remote_type_enum"
    )

    trend_enum = postgresql.ENUM(
        "weekly","monthly",
        name="trend_period_enum"
    )

    scrape_status_enum = postgresql.ENUM(
        "running","success","partial","failed",
        name="scrape_status_enum"
    )

    company_size_enum = postgresql.ENUM(
        "1-10","11-50","51-200","201-1000","1001-5000","5000+","unknown",
        name="company_size_enum"
    )

    for e in [
        job_source_enum,
        seniority_enum,
        remote_enum,
        trend_enum,
        scrape_status_enum,
        company_size_enum,
    ]:
        e.create(op.get_bind(), checkfirst=True)

    # ---------------- COMPANIES ----------------

    op.create_table(
        "companies",

        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("website", sa.String(512)),
        sa.Column("industry", sa.String(128)),
        sa.Column("size", sa.Enum(name="company_size_enum"), server_default="unknown"),
        sa.Column("hq_location", sa.String(255)),
        sa.Column("linkedin_url", sa.String(512)),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.UniqueConstraint("normalized_name", name="uq_companies_normalized_name"),
    )

    op.create_index("ix_companies_normalized_name","companies",["normalized_name"])

    # ---------------- SCRAPE RUNS ----------------

    op.create_table(
        "scrape_runs",

        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("source", sa.Enum(name="job_source_enum"), nullable=False),
        sa.Column("status", sa.Enum(name="scrape_status_enum"), server_default="running"),
        sa.Column("config", postgresql.JSONB()),

        sa.Column("jobs_found", sa.Integer(), server_default="0"),
        sa.Column("jobs_new", sa.Integer(), server_default="0"),
        sa.Column("jobs_updated", sa.Integer(), server_default="0"),
        sa.Column("jobs_skipped", sa.Integer(), server_default="0"),

        sa.Column("error_log", sa.Text()),
        sa.Column("error_count", sa.Integer(), server_default="0"),

        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )

    op.create_index("ix_scrape_runs_source","scrape_runs",["source","started_at"])
    op.create_index("ix_scrape_runs_status","scrape_runs",["status"])
    op.create_index("ix_scrape_runs_started_at","scrape_runs",["started_at"])

    # ---------------- JOB POSTINGS ----------------

    op.create_table(
        "job_postings",

        sa.Column(
            "job_posting_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("url", sa.String(1024), nullable=False),

        sa.Column("location", sa.String(255)),
        sa.Column("remote_type", sa.Enum(name="remote_type_enum"), server_default="unknown"),

        sa.Column("salary_min", sa.Numeric(12,2)),
        sa.Column("salary_max", sa.Numeric(12,2)),
        sa.Column("salary_currency", sa.String(8)),

        sa.Column("seniority", sa.Enum(name="seniority_level_enum"), server_default="unknown"),

        sa.Column("source", sa.Enum(name="job_source_enum"), nullable=False),
        sa.Column("external_id", sa.String(255)),

        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("is_processed", sa.Boolean(), server_default="false"),

        sa.Column("posted_at", sa.DateTime(timezone=True)),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("companies.company_id", ondelete="SET NULL"),
        ),

        sa.Column(
            "scrape_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("scrape_runs.id", ondelete="SET NULL"),
        ),

        sa.UniqueConstraint("source","external_id",name="uq_job_postings_source_external"),
    )

    op.create_index("ix_job_postings_posted_at","job_postings",["posted_at"])
    op.create_index("ix_job_postings_scraped_at","job_postings",["scraped_at"])
    op.create_index("ix_job_postings_source","job_postings",["source","scraped_at"])
    op.create_index("ix_job_postings_company","job_postings",["company_id"])
    op.create_index("ix_job_postings_seniority","job_postings",["seniority"])

    # ---------------- SKILL CATEGORIES ----------------

    op.create_table(
        "skill_categories",

        sa.Column(
            "skill_category_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("slug", sa.String(128), nullable=False, unique=True),
        sa.Column("description", sa.Text()),

        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skill_categories.skill_category_id", ondelete="SET NULL"),
        ),

        sa.Column("sort_order", sa.Integer(), server_default="0"),
    )

    # ---------------- SKILLS ----------------

    op.create_table(
        "skills",

        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text()),

        sa.Column(
            "category_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skill_categories.skill_category_id", ondelete="SET NULL"),
        ),

        sa.Column("is_verified", sa.Boolean(), server_default="false"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("metadata", postgresql.JSONB()),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.UniqueConstraint("normalized_name", name="uq_skills_normalized_name"),
    )

    op.create_index("ix_skills_category","skills",["category_id"])

    # ---------------- SKILL ALIASES ----------------

    op.create_table(
        "skill_aliases",

        sa.Column(
            "skill_alias_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column("alias", sa.String(255), nullable=False),

        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.UniqueConstraint("alias", name="uq_skill_aliases_alias"),
    )

    op.create_index("ix_skill_aliases_skill_id","skill_aliases",["skill_id"])

    # ---------------- JOB SKILLS ----------------

    op.create_table(
        "job_skills",

        sa.Column(
            "job_skill_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column(
            "job_posting_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job_postings.job_posting_id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column("frequency_mentioned", sa.Integer(), server_default="1"),
        sa.Column("context_snippet", sa.Text()),
        sa.Column("confidence_score", sa.Float()),
        sa.Column("is_required", sa.Boolean()),
        sa.Column("extraction_method", sa.String(64)),

        sa.UniqueConstraint("job_posting_id","skill_id",name="uq_job_skills_posting_skill"),
    )

    # ---------------- SKILL TRENDS ----------------

    op.create_table(
        "skill_trends",

        sa.Column(
            "skill_trend_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),

        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skills.skill_id", ondelete="CASCADE"),
            nullable=False,
        ),

        sa.Column("period", sa.Enum(name="trend_period_enum"), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),

        sa.Column("job_count", sa.Integer(), server_default="0"),
        sa.Column("demand_score", sa.Float()),
        sa.Column("growth_rate", sa.Float()),

        sa.Column("avg_salary_min", sa.Numeric(12,2)),
        sa.Column("avg_salary_max", sa.Numeric(12,2)),

        sa.Column("co_occurring_skills", postgresql.ARRAY(postgresql.UUID)),
        sa.Column("top_companies", postgresql.ARRAY(postgresql.UUID)),
        sa.Column("geo_distribution", postgresql.JSONB()),

        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.UniqueConstraint("skill_id","period","period_start",name="uq_skill_trends_skill_period"),
    )


def downgrade():

    op.drop_table("skill_trends")
    op.drop_table("job_skills")
    op.drop_table("skill_aliases")
    op.drop_table("skills")
    op.drop_table("skill_categories")
    op.drop_table("job_postings")
    op.drop_table("scrape_runs")
    op.drop_table("companies")