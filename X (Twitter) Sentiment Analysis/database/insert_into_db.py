# =============================================================================
# Imports
# =============================================================================
from config.settings import KAGGLE_TRAIN_DATASET
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger
from utils.text_util import content_hash
from database.connect import DatabaseConnection

# Optional visualization imports (currently not used)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Database Setup
# =============================================================================
# Initialize MongoDB connection
db = DatabaseConnection()

# Ensure unique index on content_hash to prevent duplicate tweets
db.create_index("source_tweet", "content_hash", unique=True)

# Print current number of documents in the collection
print(f"Current total tweets in DB: {db.count('source_tweet')}")

# =============================================================================
# Helper Functions
# =============================================================================
def create_metadata(row) -> dict:
    """
    Generate metadata dictionary for a tweet record.

    Args:
        row (pd.Series): Row from DataFrame containing tweet info.

    Returns:
        dict: Metadata dictionary for MongoDB insertion.
    """
    return {
        "source": "kaggle",
        "topic": row['topic'],
        "sentiment": row['sentiment'],
        "text": row['text'],
        "char_count": row['char_count'],
        "token_count": row['token_count'],
        "content_hash": row['content_hash'],
        "language": "en",
    }

# =============================================================================
# Load and Preprocess Kaggle Dataset
# =============================================================================
# Load dataset
df = pd.read_csv(KAGGLE_TRAIN_DATASET)

# Display basic info
print(df.info())

# Rename columns for clarity
df.columns = ["user_id", "topic", "sentiment", "text"]

# Remove duplicates and missing values
df = df.drop_duplicates().dropna()

# Compute text statistics
df['char_count'] = df['text'].apply(len)             # Number of characters
df['token_count'] = df['text'].str.split().str.len() # Number of tokens/words

# Add source and content hash
df['source'] = "kaggle"
df['content_hash'] = df['text'].apply(content_hash)

# Generate metadata column
df['meta_data'] = df.apply(create_metadata, axis=1)

# Drop unnecessary columns
df = df.drop(columns=['user_id', 'topic'])

# Add creation timestamp in UTC
df["created_at"] = datetime.now(timezone.utc)

# Display updated DataFrame info
print("="*40)
print(df.info())
print("="*40)

# =============================================================================
# Insert Data into MongoDB (Commented Out)
# =============================================================================
"""
logger.info("Inserting tweets into database...")
records = df.to_dict(orient="records")
db.insert_many("source_tweet", records)
logger.success("Finished inserting into DB")
"""

# =============================================================================
# Example Queries
# =============================================================================
# Count number of positive tweets
positive_count = db.count("source_tweet", {"sentiment": "Neutral"})
print(f"Number of neutral tweets: {positive_count}")
print(db.find_one("source_tweet",{"sentiment":"Positive"})["_id"])
