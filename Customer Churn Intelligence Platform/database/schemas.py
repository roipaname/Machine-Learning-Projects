from sqlalchemy import Column,Text,Integer,Float,DateTime,Boolean,Index,UniqueConstraint
from sqlachemy.ext.declarative import declarative_base
from slqalchemy.dialects.postgresql import JSON,UUID

from datetime import datetime
from  loguru import logger
import uuid


base=declarative_base()