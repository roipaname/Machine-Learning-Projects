from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.utils.extractor import extract_category
import logging
from typing import Dict
logging.basicConfig(level=logging.INFO)


class PreprocessingPipeline:
    """Orchestrate full text preprocessing."""
    def __init__(self):
        self.cleaner=TextCleaner()
        self.normalizer=TextNormalizer()


    def process(self, title: str, body: str,url:str) -> Dict:
        """Process article title and body."""
        try:
            # Clean
            clean_title = self.cleaner.clean(title, aggressive=False)
            clean_body = self.cleaner.clean(body, aggressive=True)
            
            # Normalize
            title_tokens = self.normalizer.normalize(clean_title)
            body_tokens = self.normalizer.normalize(clean_body)
            
            # Combine for classification
            all_tokens = title_tokens + body_tokens
            
            # Reconstruct as text (for TF-IDF)
            processed_text = ' '.join(all_tokens)
            
            return {
                'cleaned_title': clean_title,
                'cleaned_body': clean_body,
                'processed_text': processed_text,
                'token_count': len(all_tokens),
                'language': 'en',  # Can add language detection here
                'category':extract_category(url)
            }
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return None