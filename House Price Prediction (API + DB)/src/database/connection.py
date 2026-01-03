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
    def execute_query(self,query,params=None):
        """Execute a raw SQL query and return the results as a DataFrame."""
        if not self.engine:
            self.connect()
        try:
            with self.engine.connect() as connection:
                result=connection.execute(text(query),params or {})
                connection.commit()
                return result
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise
    def read_sql(self, query, params=None):
        """Read SQL query into DataFrame"""
        if not self.engine:
            self.connect()
        
        return pd.read_sql(query, self.engine, params=params)

    def get_table_count(self,schema,table):
        """Get the count of rows for a table"""
        query=f"SELECT COUNT(*) FROM {schema}.{table};"
        result=self.read_sql(query)
        return result.iloc[0,0]
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logging.info("Database connection closed.")

            

if __name__ == "__main__":
    db = DatabaseConnection()
    db.test_connection()
    
    # Show all tables
    query = """
    SELECT schemaname, tablename 
    FROM pg_tables 
    WHERE schemaname IN ('raw', 'processed', 'models')
    ORDER BY schemaname, tablename;
    """
    tables = db.read_sql(query)
    print("\nAvailable tables:")
    print(tables)