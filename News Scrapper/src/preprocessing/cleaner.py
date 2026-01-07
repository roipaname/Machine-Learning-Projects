import re
import unicodedata
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

class TextCleaner:
    """Remove noise from raw article text"""

    @staticmethod
    def remove_html(text:str)->str:
        """Strip any remaining HTML tags"""
        soup=BeautifulSoup(text,'html.parser')
        return soup.get_text(separator=" ")
    @staticmethod
    def fix_encoding(text:str)->str:
        """Handle encoding issues"""
        text=unicodedata.normalize('NFKD',text)
        text=text.encode('ascii','ignore').decode('ascii')
        return text

    @staticmethod
    def remove_urls(text:str)->str:
        """Remove URLS"""
        return re.sub(r'http\S+|www.\S+','',text)
    
    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = False) -> str:
        """Remove special characters."""
        if keep_punctuation:
            # Keep basic punctuation for sentence structure
            pattern = r'[^a-zA-Z0-9\s.,!?;:\'-]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Normalize whitespace."""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove standalone numbers."""
        return re.sub(r'\b\d+\b', '', text)
    
    def clean(self,text:str, aggressive:bool)->str:
        """Full cleaning pipeline"""
        if not text:
            return ""
        text=self.remove_html(text)
        text=self.fix_encoding(text)
        text=self.remove_urls(text)

        if aggressive:
            text=self.remove_numbers(text)
            text=self.remove_special_chars(text,keep_punctuation=False)
        else:
            text=self.remove_special_chars(text,keep_punctuation=True)

        text=self.remove_extra_whitespace(text)
        text=text.lower()

        return text
