from datetime import datetime
from typing import List,Dict,Optional
from bs4 import BeautifulSoup
import hashlib
from dateutil import parser as date_parser


from src.scraper.base_scraper import BaseScraper
from src.utils.extractor import extract_category
from config.settings import get_scraper_config

class ArticleScraper(BaseScraper):
    """Article Scraper implementation from base scraper"""

    @staticmethod
    def extract_text(soup:BeautifulSoup,selector:str)->Optional[str]:
        element = soup.select_one(selector)
        return element.get_text(strip=True) if element else None
    
    @staticmethod
    def extract_all_text(soup:BeautifulSoup,selector:str)->str:
        elements=soup.select(selector)
        return " ".join(el.get_text(strip=True) for el in elements) if elements else ""
    @staticmethod
    def parse_date(date_string:Optional[str])->Optional[datetime]:
        if not date_string:
            return None
        try:
            return date_parser.parse(date_string)
        except Exception:
            return None
    @staticmethod
    def generate_content_hash(title:str,body:str)->str:
        content=f"{title}{body}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def parse_article(self, html:str, url:str)->Dict:
        soup=BeautifulSoup(html,'html.parser')

        title=self.extract_text(soup,self.selectors['title'])
        body=self.extract_all_text(soup,self.selectors['body'])
        date_str=self.extract_text(soup,self.selectors.get("date","time"))
        
        if not title or not body or len(body)<50:
            return None
        published_date=self.parse_date(date_str)

        return {
            "url": url,
            "title": title,
            "body": body,
            "published_date": published_date,
            "scraped_at": datetime.utcnow(),
            "content_hash": self.generate_content_hash(title, body),
            "source": self.name,
            
        }
if __name__ == "__main__":
    import yaml
    import logging
    from pathlib import Path
    import sys

    logging.basicConfig(level=logging.INFO)

    # ---------------------------------------------------------------------
    # Ensure project root is on PYTHONPATH
    # ---------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    # ---------------------------------------------------------------------
    # Load source configuration
    # ---------------------------------------------------------------------
    config_path = PROJECT_ROOT / "config" / "sources.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", [])
    if not sources:
        raise ValueError("No sources found in sources.yaml")

    # ---------------------------------------------------------------------
    # Pick first source for testing
    # ---------------------------------------------------------------------
    source_config = sources[1]
    scraper_config=get_scraper_config()

    headers = scraper_config['headers']

    scraper = ArticleScraper(
        source_config=source_config,
        headers=headers
    )

    # ---------------------------------------------------------------------
    # Run test scrape
    # ---------------------------------------------------------------------
    logging.info(f"Testing scraper for: {scraper.name}")

    articles = scraper.scrape(max_articles=3)

    # ---------------------------------------------------------------------
    # Output results
    # ---------------------------------------------------------------------
    print("\n================ SCRAPE RESULT ================\n")

    for i, article in enumerate(articles, 1):
        print(f"Article {i}")
        print("-" * 50)
        print(f"Title   : {article['title']}")
        print(f"URL     : {article['url']}")
        print(f"Date    : {article['published_date']}")
        print(f"Hash    : {article['content_hash']}")
        print(f"Source  : {article['source']}")
        print(f"Preview : {article['body'][:200]}...\n")

    print(f"Total articles scraped: {len(articles)}")
