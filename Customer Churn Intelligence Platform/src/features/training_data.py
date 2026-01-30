import pandas as pd
from src.features.feature_eng import extract_all_customers_features
from database.connection import DatabaseConnection
from database.schemas import Churn_Labels
from loguru import logger
from pathlib import Path
from config.settings import DATA_PROCESSED_DIR

db=DatabaseConnection()

def build_training_dataset()->pd.DataFrame:
    """
    Combine features with churn labels to create ML-ready dataset"""

    features_df=extract_all_customers_features()
    logger.info(f"Extracted features for {len(features_df)} Customers")

    with db.get_db() as session:
        labels=session.query(Churn_Labels).all()
        labels_df=pd.DataFrame([{
            'customer_id':label.customer_id,
            'churned':label.churned,
            'churn_date':label.churn_date

        }] for label in labels)

        training_df=features_df.merge(labels_df,on='customer_id',how='left')


        # Fill missing churn labels (customers who haven't churned)
        training_df['churned']=training_df['churned'].fillna(False)

        logger.success(f"Built training dataset with {len(training_df)} records")
        # Save to parquet (feature store simulation)
        training_df.to_parquet(Path(DATA_PROCESSED_DIR)/'training_features.parquet',index=False)

        return training_df