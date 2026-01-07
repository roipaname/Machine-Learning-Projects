from abc import ABC,abstractmethod
from typing import List,Dict,Optional
from src.utils.validators import is_valid_url
import requests
from bs4 import BeautifulSoup
import time
from database.operations import article_exists_by_url
import logging
logging.basicConfig(level=logging.INFO)

class BaseScraper(ABC):
    """Base Class for all news scraper"""

    def __init__(self,source_config:Dict,headers:Dict):
        self.name=source_config['name']
        self.base_url=source_config['url']
        self.selectors=source_config['selector']
        self.rate_limit=source_config.get('rate_limit',2)
        self.headers=headers
        self.session=requests.Session()
        self.session.headers.update(headers)

    def fetch_page(self,url:str,max_retries:int=3)->Optional[str]:
        """Fetch page content with retries and exponential backoff"""
        for attempt in range(max_retries):
            try:
                response=self.session.get(url,timeout=10)
                response.raise_for_status()
                time.sleep(self.rate_limit)
                return response.text
            except Exception as e:
                wait_time= 2**attempt
                logging.error(f"Attempt {attempt+1} failed for {url} : {e}")
                if attempt<max_retries-1:
                    time.sleep(wait_time)
                else:
                    logging.error(f"failed to fetch {url} after {max_retries}")
                    return None
    def extract_article_links(self, html:str)->List[str]:
        """Extract article URLs from listing page"""
        soup=BeautifulSoup(html,'html.parser')
        links=[]
        for link in soup.select(self.selectors['article_links']):
            href=link.get('href')
            if href:
                # Handle relative URLS
                if href.startswith('/'):
                    href=f"{self.base_url.rstrip('/')}{href}"
                if is_valid_url(href):
                    links.append(href)
        return list(set(links)) #Removes dduplicates
    @abstractmethod
    def parse_article(self,html:str,url:str)->Dict:
        """Extract article Content . Must be implemnented by subclasses"""
        pass
    def scrape(self,max_articles:int=50)->List[Dict]:
        """Main scraping logic"""
        logging.info(f"Starting scrape for {self.name}")

        listing_html=self.fetch_page(self.base_url)
        if not listing_html:
            return []
        
        # Get article URLs
        article_urls=self.extract_article_links(listing_html)
        logging.info(f"Found {len(article_urls)} article links")

        articles=[]

        for url in article_urls[:max_articles]:

            # Skip if already in databse
            if article_exists_by_url(url):
                logging.info(f"Skipping duplicate: {url}")
                continue

            article_html=self.fetch_page(url)
            if article_html:
                article_data= self.parse_article(article_html,url)
                if article_data:
                    articles.append(article_data)
                    logging.info(f"Scraped : {article_data['title'][:50]}...")
        logging.info(f"Scraped {len(articles)} new articles from {self.name}")
        return articles

        