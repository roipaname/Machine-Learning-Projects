from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,Session
from config.settings import (
    DB_NAME,DB_HOST,DB_PASSWORD,DB_PORT,DB_USER,DB_POOL_SIZE,DB_MAX_OVERFLOW,DB_ECHO
)

from loguru import logger
from contextlib import contextmanager



class DatabaseConnection:

    def __init__(self):
        self.localSession=None
        self.db_url=f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.engine=self.init_db()

    def init_db(self):
        """Create a database connection and session."""
        try:
            self.engine=create_engine(self.db_url,pool_size=DB_POOL_SIZE,max_overflow=DB_MAX_OVERFLOW,echo=DB_ECHO,pool_pre_ping=True)
            self.localSession=sessionmaker(bind=self.engine,expire_on_commit=False)
            from database.schemas import Base
            Base.metadata.create_all(bind=self.engine)
            logger.success("Database Initialized")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to Initialize DB:{e}")
            raise

    @contextmanager
    def get_db(self):
        session=self.localSession()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database Error : {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    import uuid
    from datetime import datetime
    from database.schemas import Accounts  
    from loguru import logger

    # Initialize the database connection
    db = DatabaseConnection()

    try:
        with db.get_db() as session:
            logger.info("Database session opened successfully")

            test_account=Accounts(
                account_id=uuid.uuid4(),
                company_name="Paname LTD",
                industry="Entertainment",
                company_size=5,
                contract_type="monthly",
                account_tier="gold"

            )
            session.add(test_account)
            session.flush()
            logger.info(f"Inserted test article with ID: {test_account.account_id}")
            result=session.query(
                Accounts
            ).filter_by(
                account_id=test_account.account_id
            ).first()
            if result:
                logger.info(f"Successfully retrieved account: {result.account_id}")
            else:
                logger.error("Failed to retrieve the test account")

    except Exception as e:
        logger.error(f"Test failed: {e}")