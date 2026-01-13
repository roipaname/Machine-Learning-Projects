from src.preprocessing.pipeline import PreprocessingPipeline
from database.connect import DatabaseConnection
from loguru import logger
from typing import List, Dict
import time

def process_tweets_batch(tweets: List[Dict], pipeline: PreprocessingPipeline, db: DatabaseConnection) -> int:
    """Process a batch of tweets and insert them into the database.
    
    Args:
        tweets: List of source tweet documents
        pipeline: Preprocessing pipeline instance
        db: Database connection instance
        
    Returns:
        Number of successfully processed tweets
    """
    processed_docs = []
    
    for tweet in tweets:
        try:
            # Process the tweet text through the pipeline
            processed_data = pipeline.process(tweet['text'])
            
            # Combine processed data with metadata
            processed_doc = {
                **processed_data,
                "source_id": tweet["_id"],
                "sentiment": tweet["sentiment"]
            }
            processed_docs.append(processed_doc)
            
        except Exception as e:
            logger.error(f"Failed to process tweet {tweet['_id']}: {e}")
            continue
    
    # Bulk insert all processed documents at once
    if processed_docs:
        db.insert_many("processed_tweet", processed_docs)
        logger.debug(f"Inserted batch of {len(processed_docs)} processed tweets")
        print(f"Inserted into Processed_tweet:{db.count('processed_tweet')}")
    
    return len(processed_docs)


def main():
    """Main function to process and insert all source tweets."""
    start_time = time.time()
    logger.info("Starting processed tweet insertion")
    
    # Initialize connections
    db = DatabaseConnection()
    pipeline = PreprocessingPipeline()
    
    # Fetch only necessary fields to reduce memory usage
    logger.info("Fetching source tweets from database")
    source_tweets = db.find_many(
        collection="source_tweet",
        projection={"_id": 1, "text": 1, "sentiment": 1}
    )
    
    total_tweets = len(source_tweets)
    logger.info(f"Found {total_tweets} source tweets to process")
    
    if total_tweets == 0:
        logger.warning("No source tweets found. Exiting.")
        return
    
    # Process in batches for better performance
    batch_size = 500
    total_processed = 0
    
    for i in range(0, total_tweets, batch_size):
        batch = source_tweets[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_tweets + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tweets)")
        
        try:
            processed_count = process_tweets_batch(batch, pipeline, db)
            total_processed += processed_count
            
            # Log progress
            progress_pct = (total_processed / total_tweets) * 100
            logger.info(f"Progress: {total_processed}/{total_tweets} ({progress_pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {e}")
            continue
    
    # Log completion summary
    elapsed_time = time.time() - start_time
    tweets_per_sec = total_processed / elapsed_time if elapsed_time > 0 else 0
    
    logger.success(f"Completed processing {total_processed}/{total_tweets} tweets")
    logger.info(f"Total time: {elapsed_time:.2f}s ({tweets_per_sec:.1f} tweets/sec)")
    
    if total_processed < total_tweets:
        logger.warning(f"Failed to process {total_tweets - total_processed} tweets")


if __name__ == "__main__":
   # main()