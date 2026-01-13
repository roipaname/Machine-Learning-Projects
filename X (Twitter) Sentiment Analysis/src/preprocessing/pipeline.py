from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
import logging
from typing import Dict
from datetime import datetime,timezone

logging.basicConfig(level=logging.INFO)


class PreprocessingPipeline:
    def __init__(self):
        self.cleaner=TextCleaner()
        self.normalizer=TextNormalizer()

    def process(self,text:str)->Dict:

        cleaned_text=self.cleaner.clean(text)
        normalized_tokens=self.normalizer.normalize(cleaned_text)

        return {
            "token_count":len(normalized_tokens),
            "processed_text":"".join(normalized_tokens),
            "processed_at":datetime.now(timezone.utc),
            "updated_at":datetime.now(timezone.utc),
            "language":self.normalizer.langauage

        }