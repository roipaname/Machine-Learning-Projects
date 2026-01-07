from database.connection import DatabaseConnection
import logging
from typing import List,Optional,Dict
from sqlalchemy.exc import IntegrityError
from database.models import SourceArticle,ProcessedArticle
import uuid

logging.basicConfig(level=logging.INFO)
db=DatabaseConnection()

def insert_raw_articles(article_data:Dict)->Optional[uuid.UUID]:
    """Insert article , return ID if successful"""
    with db.get_db() as session:
        try:
            if not article_exists_by_url(article_data['url']):
                article=SourceArticle(**article_data)
                session.add(article)
                session.flush()
                logging.info(f"Inserted article ID :{article.id}")
                return article.id

            
        except Exception as e:
            logging.error(f"failed to create raw article :{e}")
            raise

def article_exists_by_url(url:str)->bool:
    """Chexk if URL already scraped"""
    with db.get_db() as session:
        return session.query(SourceArticle).filter_by(url=url).first() is not None

def article_exists_by_hash(content_hash:str)->bool:
    with db.get_db() as session:
        return session.query(SourceArticle).filter_by(content_hash=content_hash).first() is not None
def get_unprocessed_articles(limit:int=100)->List[SourceArticle]:
    """Get a list of unpprocessed articles"""
    with db.get_db() as session:
        return session.query(SourceArticle).filter(
            ~SourceArticle.id.in_(
                session.query(ProcessedArticle.source_article_id)
            )
        ).limit(100).all()
def insert_processed_article(processed_data:Dict)->Optional[uuid.UUID]:
    """Store Processed article."""
    with db.get_db() as session:
        try:
            processed=ProcessedArticle(**processed_data)
            session.add(processed)
            session.flush()
            return processed.id
        except Exception as e:
            logging.error(f"Failed to insert processed : {e}")
            raise
def get_processed_articles(limit:int=100)->List[ProcessedArticle]:
    with db.get_db() as session:
        return session.query(ProcessedArticle).filter(ProcessedArticle.processed_text.isnot(None)).limit(100).all()


        

    

    
if __name__ == "__main__":
    from datetime import datetime

    logging.info("Starting database test...")

    # 1️⃣ Prepare test raw article data
    raw_article_data = {
        "url": "https://example.com/test-article5",
        "title": "Test Article",
        "body": "This is a test article content.",
        "source": "Example Source",
        "content_hash": str(uuid.uuid4()),
        "scraped_at": datetime.utcnow(),
        "extra_metadata": {"test": True}  # matches your model column
    }

    # 2️⃣ Insert raw article
    try:
        raw_id = insert_raw_articles(raw_article_data)
        logging.info(f"Raw article inserted with ID: {raw_id}")
    except Exception as e:
        logging.error(f"Failed to insert raw article: {e}")

    # 3️⃣ Check existence by URL
    exists_url = article_exists_by_url(raw_article_data["url"])
    logging.info(f"Article exists by URL? {exists_url}")

    # 4️⃣ Check existence by content hash
    exists_hash = article_exists_by_hash(raw_article_data["content_hash"])
    logging.info(f"Article exists by content hash? {exists_hash}")

    # 5️⃣ Insert processed article
    processed_article_data = {
        "source_article_id": raw_id,
        "cleaned_title": "Cleaned Test Article",
        "cleaned_body": "This is cleaned content for testing.",
        "token_count": 10,
        "language": "en",
        "is_duplicate": False,
        "processed_at": datetime.utcnow()
    }

    try:
        processed_id = insert_processed_article(processed_article_data)
        logging.info(f"Processed article inserted with ID: {processed_id}")
    except Exception as e:
        logging.error(f"Failed to insert processed article: {e}")

    # 6️⃣ Retrieve unprocessed articles (should be empty now)
    unprocessed = get_unprocessed_articles(limit=10)
    logging.info(f"Unprocessed articles retrieved: {len(unprocessed)}")
    if unprocessed:
        for art in unprocessed:
            logging.info(f"- {art.title}")
    else:
        logging.info("No unprocessed articles found.")
