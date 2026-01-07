import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import os
import sys
import logging
import json

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import DatabaseConnection
logging.basicConfig(level=logging.INFO)
class FeatureEngineer:
    """Engineer Features from raw property data."""
    def __init__(self):
        self.db=DatabaseConnection()
        self.db.connect()
        self.encoders={}
    def extract_raw_data(self):
        """EXTract raw data from db"""
        query=f"SELECT * FROM raw.properties WHERE price is NOT NULL;"
        df=self.db.read_sql(query)
        logging.info(f" Extracted {len(df)} rows from database.")
        return df
    def create_numerical_features(self,df:pd.DataFrame)->pd.DataFrame:
        """Create numerical features"""
        print("Creating numerical features...")
        features=pd.DataFrame()
        features['property_id']=df['property_id']
        features['total_rooms']=(df['bed'].fillna(0)+df['bath'].fillna(0))
        features['price_per_sqft']=np.where(
            df['house_size']>0,
            df['price']/df['house_size'],
            None
        )
        features['lot_sqft'] = df['acre_lot'] * 43560
        features['size_to_lot_ratio']=np.where(
            features['lot_sqft']>0,
            df['house_size']/features['lot_sqft'],
            None
        )
        features['bed_bath_ratio']=np.where(
            df['bath']>0,
            df['bed']/df['bath'],
            None
        )
        features['has_prev_sale']=df['prev_sold_date'].notna()
        features['is_large_house']=(df['house_size']>df['house_size'].median())
        logging.info("Numerical features created.")
        return features

    def create_categorical_features(self,df:pd.DataFrame,features:pd.DataFrame,save_dir='encoders')->pd.DataFrame:
        """Encode categorical features"""
        os.makedirs(save_dir, exist_ok=True)
        cat_cols={col:f"{col}_encoded" for col in ['state','city','status']}
        for col,encoded_name in cat_cols.items():
            if col in df.columns:
                unique_values=df[col].dropna().unique()
                mapping={val:idx for idx,val in enumerate(sorted(unique_values))}
                self.encoders[col]=mapping

                json_path = os.path.join(save_dir, f"{col}_mapping.json")
                with open(json_path, "w") as f:
                    json.dump(mapping, f,indent=4)
                logging.info(f"Saved {col} mapping to {json_path}")

                features[encoded_name]=df[col].map(mapping)
                logging.info(f"Encoded {col} into {encoded_name}.")
        return features
    def create_price_categories(self,df:pd.DataFrame,features:pd.DataFrame)->pd.DataFrame:
        """Create price categories"""
        logging.info("Creating price categories...")
        price=df['price']
        q1=price.quantile(0.25)
        q2=price.quantile(0.5)
        q3=price.quantile(0.75)

        conditions=[
            price<=q1,
            (price>q1) & (price<=q2),
            (price>q2) & (price<=q3),
            price>q3
        ]
        categories=['low','medium','high','luxury']
        features['price_category']=np.select(conditions,categories,default='unknown')
        print(f" Price Ranges:")
        print(f"  Low: <= {q1:.2f}")
        print(f"  Medium: <= {q2:.2f}")
        print(f"  High: <= {q3:.2f}")
        print(f"  Luxury: > {q3:.2f}")

        return features
    def engineer_features(self)->pd.DataFrame:
        """Run Complete feature engineering pipeline"""
        print(f"\n{'='*60}")
        logging.info("Starting feature engineering pipeline...")
        print(f"{'='*60}\n")

        raw_df=self.extract_raw_data()
        features_df=self.create_numerical_features(raw_df)
        features_df=self.create_categorical_features(raw_df,features_df)
        features_df=self.create_price_categories(raw_df,features_df)

        print(f"\nFeature engineering complete!")
        print(f"  Total features created: {len(features_df.columns) - 1}")  # -1 for property_id
        print(f"  Feature names: {', '.join(features_df.columns[1:])}")
        
        return features_df
    def save_features(self, features_df: pd.DataFrame):
        """Save engineered features to database"""
        logging.info("Saving engineered features to database...")
        
        try:
            # Clear existing features first
            self.db.execute_query("TRUNCATE TABLE processed.features")
            
            # Insert new features
            features_df.to_sql(
                'features',
                self.db.engine,
                schema='processed',
                if_exists='append',
                index=False,
                method=None,
                chunksize=1000
            )
            logging.info("Engineered features saved successfully.")
            count=self.db.get_table_count('processed','features')
            logging.info(f"Total records in 'processed.features' table: {count}")
        except Exception as e:
            logging.error(f"Error saving features to database: {e}")
            raise
    def get_feature_statistics(self):
        """Display statistics about engineered features"""
        query = """
        SELECT 
            COUNT(*) as total_records,
            AVG(price_per_sqft) as avg_price_per_sqft,
            AVG(total_rooms) as avg_total_rooms,
            AVG(bed_bath_ratio) as avg_bed_bath_ratio,
            SUM(has_prev_sale::int) as properties_with_prev_sale,
            SUM(is_large_house::int) as large_houses
        FROM processed.features
        """
        stats = self.db.read_sql(query)
        print("\nFeature Statistics:")
        print(stats.T)
        
        # Price category distribution
        query = """
        SELECT price_category, COUNT(*) as count
        FROM processed.features
        GROUP BY price_category
        ORDER BY count DESC
        """
        categories = self.db.read_sql(query)
        print("\nPrice Category Distribution:")
        print(categories)


if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Engineer features
    features = engineer.engineer_features()
    
    # Save to database
    engineer.save_features(features)
    
    # Show statistics
    engineer.get_feature_statistics()
    
    print(f"\n{'='*60}")
    print(f"Feature Engineering Complete!")
    print(f"{'='*60}\n")
        
        