from database.operations import get_unprocessed_articles,insert_processed_article
from src.preprocessing.pipeline import PreprocessingPipeline
import logging

logging.basicConfig(level=logging.INFO)

def main(batch_size:int=100):
    """process raw -articles"""
    pipeline=PreprocessingPipeline()
    while True:
        raw_articles=get_unprocessed_articles()
        if not raw_articles:
            logging.info("No More articles to get")
            break
        logging.info(f"Processing {len(raw_articles)} articles")

        for article in raw_articles:
            result=pipeline.process(article.title,article.body,article.url)
            if not result:
                logging.warning(f"Failed to process {article.id}")
                continue
            processed_data={
                'source_article_id':article.id,
                **result
            }
            insert_processed_article(processed_data)
            logging.info(f"processed article ID {article.id}")
        logging.info(f"Batch complete. Processed {len(raw_articles)} articles")


if __name__=="__main__":
    main()
