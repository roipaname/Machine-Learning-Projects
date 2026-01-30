from src.features.feature_eng import extract_customer_features,extract_all_customers_features
from src.features.training_data import build_training_dataset

from loguru import logger


if __name__=="__main__":
    logger.info("Testing dataset builder")

    features_df=extract_all_customers_features()
    print(features_df.head())
    print(features_df.info())

    # Build training dataset
    training_df = build_training_dataset()
    print(f"\nChurn rate: {training_df['churned'].mean():.2%}")
