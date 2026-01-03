from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()


class DatabaseConnection:
    def __init__(self):
        self.db_user=os.getenv("DB_USER")
        self.db_password=os.getenv("DB_PASSWORD")
        self.db_host=os.getenv("DB_HOST")
        self.db_port=os.getenv("DB_PORT")
        self.db_name=os.getenv("DB_NAME")
        self.db_url=f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine=None
        self.session=None
    def connect(self):
        """Create a database connection and session."""
        try:
            self.engine=create_engine(self.db_url)
            self.session=sessionmaker(bind=self.engine)
            logging.info("Database connection established successfully.")
            return self.engine
        except Exception as e:
            logging.error(f"Error connecting to the database: {e}")
            raise
    def get_session(self):
        """Get a new Session"""
        if not self.session:
            self.connect()
        return self.session()
    def test_connection(self):
        """Test the database connection."""
        if not self.engine:
            self.connect()
        try:
            with self.engine.connect() as connection:
                result=connection.execute(text("SELECT version();"))
                version=result.fetchone()[0]
                print(f"✓ PostgreSQL Version: {version}")
                return True
        except Exception as e:
            logging.error(f"Database connection test failed: {e}")
            return False

