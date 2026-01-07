from sqlalchemy import Column,Integer,Float,String,ForeignKey,Text,DateTime,Boolean,Index,UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB,UUID

from datetime import datetime
import logging
import uuid

logging.basicConfig(level=logging.INFO)

Base=declarative_base()

class SourceArticle(Base):
    """Immmutable Scrapped Articles"""
    __tablename__='source_articles'

    id=Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)
    url=Column(String(2048),unique=True,nullable=False)
    content_hash=Column(String(64),unique=True,index=True)
    title=Column(Text,nullable=False)
    body=Column(Text,nullable=False)
    source=Column(String(255),index=True,nullable=False)
    published_date=Column(DateTime,nullable=True)
    scraped_at=Column(DateTime,nullable=False,default=datetime.utcnow)
    extra_metadata=Column(JSONB,default={})
    category=Column(Text,nullable=True)

    #Constraints
    __table_args__=(
        Index('idx_scraped_at','scraped_at'),
        Index('idx_content_hash','content_hash')
    )

class ProcessedArticle(Base):
    """Processed Article Cleaned and normalized articles"""
    __tablename__='processed_articles'

    id=Column(UUID(as_uuid=True),default=uuid.uuid4,nullable=False,primary_key=True)
    source_article_id=Column(UUID(as_uuid=True),ForeignKey('source_articles.id'),unique=True,nullable=False)
    cleaned_title=Column(Text,nullable=False)
    cleaned_body=Column(Text,nullable=False)
    token_count=Column(Integer,default=0)
    language=Column(String(200),default="en")
    category=Column(Text,nullable=False)
    processed_text=Column(Text,nullable=True)
    is_duplicate=Column(Boolean,default=False)
    duplicate_of=Column(UUID(as_uuid=True),ForeignKey('processed_articles.id'),nullable=True)
    processed_at=Column(DateTime,default=datetime.utcnow, nullable=False)

    __table_args__=(
        Index('idx_duplicate','is_duplicate'),
    )

class ArticleClassification(Base):
    """ML predictions for articles"""
    __tablename__="classified_articles"
    id=Column(UUID(as_uuid=True),nullable=False,primary_key=True,unique=True)
    processed_article_id=Column(UUID,ForeignKey('processed_articles.id'),nullable=False)
    source_article_id=Column(UUID,ForeignKey('source_articles.id'),nullable=True)

    predicted_topic=Column(String(100),nullable=False)
    confidence_score=Column(Float,nullable=False)
    model_version=Column(String(50),nullable=False)
    model_name=Column(String(100),nullable=False)
    classified_at=Column(DateTime,default=datetime.utcnow,nullable=False)

    __table_args__=(
        Index('idx_topic','predicted_topic'),
        Index('idx_model_version','model_version'),
    )
