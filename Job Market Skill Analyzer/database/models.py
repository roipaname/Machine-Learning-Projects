from sqlalchemy import (
    Column,Integer, Float,Text,DateTime,Boolean,Index,ForeignKey,Enum,
    Numeric,String,UniqueConstraint,func,BigInteger
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postsegrl import JSON,UUID,ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum
from datetime import datetime
from loguru import logger
import uuid

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