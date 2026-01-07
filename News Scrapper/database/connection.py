from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,Session
from contextlib import contextmanager
from config.settings import DATABASE_URL
from loguru import logger
import logging

logging.basicConfig(level=logging.INFO)


class DatabaseConnection:
    def __init__(self):
        self.sessionLocal=None
        self.engine=self.init_db()
        
    
    def init_db(self):
        try:
            engine=create_engine(
                DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            self.sessionLocal=sessionmaker(bind=engine,expire_on_commit=False)
            from database.models import Base
            Base.metadata.create_all(bind=engine)
            logging.info("Database initialized")

            return  engine
        except Exception as e:
            logging.info("Database not initialized: {e}")
            raise
    @contextmanager
    def get_db(self)->Session:
        """Context Manger for database sessions"""
        session=self.sessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database Error : {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    import uuid
    from datetime import datetime
    from database.models import SourceArticle  # or your ORM class
    from loguru import logger

    # Initialize the database connection
    db = DatabaseConnection()

    # Test the connection and tables
    try:
        with db.get_db() as session:
            logging.info("Database session opened successfully")

            # Insert a test article
            test_article = SourceArticle(
                url="https://example.com/test-article7",
                title="Test Article",
                body="This is a test article content.",
                source="Example Source",
                content_hash=str(uuid.uuid4()),  # unique hash for testing
                scraped_at=datetime.utcnow(),
                extra_metadata={"test": True}
            )
            session.add(test_article)
            session.flush()  # assign UUID
            logging.info(f"Inserted test article with ID: {test_article.id}")

            # Query back to check
            result = session.query(SourceArticle).filter_by(
                id=test_article.id
            ).first()
            if result:
                logging.info(f"Successfully retrieved article: {result.title}")
            else:
                logging.error("Failed to retrieve the test article")

    except Exception as e:
        logging.error(f"Test failed: {e}")
