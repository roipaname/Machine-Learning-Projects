
import yaml
import logging
from pathlib import Path
import sys
from config.settings import get_scraper_config

logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.article_parser import ArticleScraper
from database.operations import insert_raw_articles

config_path = PROJECT_ROOT / "config" / "sources.yaml"
if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
        config = yaml.safe_load(f)
sources = config.get("sources", [])
if not sources:
        raise ValueError("No sources found in sources.yaml")
scraper_config=get_scraper_config()
headers = scraper_config['headers']
def main():
    for source_config in sources:
           scraper=ArticleScraper(source_config,headers)
           logging.info(f"Scraping from {source_config['url']}")
           articles=scraper.scrape(max_articles=200)
           if not articles:
                 continue
           
           
           for article in articles:
                try:
                    insert_raw_articles(article)
                    logging.info(f"article saved in db successfully:{article['title']}")
                except Exception as e:
                      logging.error(f"failed to insert {article['title']}")
                
if __name__ == "__main__":
      main()           