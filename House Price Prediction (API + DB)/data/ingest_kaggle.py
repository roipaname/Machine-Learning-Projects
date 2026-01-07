import pandas as pd

from dotenv import load_dotenv
import os
import sys
import json
from datetime import datetime
import logging
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import DatabaseConnection
load_dotenv()
logging.basicConfig(level=logging.INFO)
class KaggleDataIngestor:
    """Ingest Kaggle dataset into PostgreSQL database."""
    def __init__(self):
        self.db=DatabaseConnection()
        self.db.connect()
    def clean_data(self,df:pd.DataFrame):
        """Clean and prepare data for ingestion"""
        df_clean=df.copy()
        #Handling missing values by filling with 'Unknown' for categorical and median for numerical
        if 'prev_sold_date' in df_clean.columns:
            df_clean['prev_sold_date']=pd.to_datetime(df_clean['prev_sold_date'],errors='coerce')
        
        if 'zip_code' in df_clean.columns:
           df_clean['zip_code'] = (
           df_clean['zip_code'].astype(str).str.replace('.0', '', regex=False).str.zfill(5)
          )

        
        num_cols=df_clean.select_dtypes(include=['number']).columns

        for col in num_cols:
            if col in df_clean.columns and col !='zip_code':
                df_clean[col]=pd.to_numeric(df_clean[col],errors='coerce')
                median_value=df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)

        cat_cols=df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in df_clean.columns and col != 'prev_sold_date'  :
                df_clean[col]=df_clean[col].astype(str).fillna('Unknown')
        initial_count=len(df_clean)
        df_clean=df_clean[df_clean['price'].notna()]
        removed=initial_count-len(df_clean)
        if removed>0:
            logging.info(f"Removed {removed} rows with missing price values.")

        #Remove extreme outliers in price
        Q1=df_clean['price'].quantile(0.01)
        Q3=df_clean['price'].quantile(0.99)
        df_clean =df_clean[
            df_clean['price'].between(Q1,Q3)
        ]
        logging.info(f"Data cleaned. Final dataset rows: {len(df_clean)}")
        return df_clean
    
    def transform_to_schema(self,df:pd.DataFrame)->pd.DataFrame:
        """Transform DataFrame to match database schema"""
        logging.info("Transforming DataFrame to match database schema...")
        properties=[]

        for idx , row in df.iterrows():
            raw={
                k: (None if pd.isna(v) else v) for k,v in row.to_dict().items()
            }
            property_dict={
                'property_id': f"{uuid.uuid4()}",
                'brokered_by':int(row.get('brokered_by')) if pd.notna(row.get('brokered_by')) else None,
                'status':row.get('status',''),
                'price':float(row.get('price')),
                'bed':int(row.get('bed')) if pd.notna(row.get('bed')) else None,
                'bath':float(row.get('bath')) if pd.notna(row.get('bath')) else None,
                'acre_lot':float(row.get('acre_lot')) if pd.notna(row.get('acre_lot')) else None,
                'street':row.get('street'),
                'city':str(row.get('city')),
                'state':str(row.get('state')),
                'zip_code':str(row.get('zip_code','')),
                'house_size':float(row.get('house_size')) if pd.notna(row.get('house_size')) else None,
                'prev_sold_date':row.get('prev_sold_date') if pd.notna(row.get('prev_sold_date')) else None,
                'source':'Kaggle',
                'raw_json':json.dumps(raw,default=str),

            }
            properties.append(property_dict)
        logging.info("Transformation complete.")
        return pd.DataFrame(properties)
    
    def ingest(self,csv_path:str,batch_size:int=1000):
        """Ingest data from CSV to database"""
        print(f"\n{'='*60}")
        logging.info(f"Starting data ingestion from {csv_path}...")

        logging.info(f"Reading CSV file from {csv_path}...")
        df=pd.read_csv(csv_path)
        logging.info(f"CSV file read successfully. Total rows: {len(df)}")
        df_clean=self.clean_data(df)
        #transformed data
        properties=self.transform_to_schema(df_clean)

        #Insert into databse in batches
        logging.info("Ingesting data into database...")
        total_inserted=0
        try:
            properties.to_sql('properties',self.db.engine,schema='raw',if_exists='append',index=False,method=None,chunksize=batch_size)
            total_inserted=len(properties)
            logging.info(f"Data ingestion complete. Total records inserted: {total_inserted}")

            #Verify Count
            db_count=self.db.get_table_count('raw','properties')
            logging.info(f"Total records in 'raw.properties' table: {db_count}")
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise
        finally:
            self.db.close()
            logging.info("Database connection closed.")
            print(f"{'='*60}\n")
            return total_inserted
    def show_sample_data(self,limit=15):
        """Display Sameple data from database"""
        query=f"""
        SELECT property_id,city,state,status,price,bed,bath,house_size FROM raw.properties ORDER BY created_at DESC LIMIT {limit};
        """
        df=self.db.read_sql(query)
        print(f"\n{'='*50}")
        print("Sample Data from 'raw.properties' table:")
        print(df)
        return df


if __name__ == "__main__":
    # Run ingestion
    ingestor = KaggleDataIngestor()
    
    # Path to your CSV
    csv_path = os.getenv("PATH_TO_DATASET", "data/raw/kaggle_house_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        print("Please place your CSV file in data/raw/ directory")
    else:
        # Ingest data
        count = ingestor.ingest(csv_path)
        
        # Show sample
        ingestor.show_sample_data()
        
            