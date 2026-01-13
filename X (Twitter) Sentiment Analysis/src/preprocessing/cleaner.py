import re
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)


class TextCleaner:
    @staticmethod
    def fix_encoding(text:str)->str:
        text=unicodedata.normalize("NFKD",text)
        text=text.encode("ascii",'ignore').decode('ascii')
        return text
    @staticmethod
    def remove_urls(text:str)->str:
        text=re.sub(r'http\S+|www.\S+','',text)
        return text
    @staticmethod
    def remove_special_chars(text:str,keep_punctuation:bool=False)->str:
        if keep_punctuation:
            pattern=pattern = r'[^a-zA-Z0-9\s.,!?;:\'-]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern,'',text)
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Normalize whitespace."""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove standalone numbers."""
        return re.sub(r'\b\d+\b', '', text)
    

    def clean(self,text:str,aggressive:bool=False)->str:
        if not text:
            return ""
        text=self.fix_encoding(text)
        text=self.remove_urls(text)
        if aggressive:
            text=self.remove_numbers(text)
            text=self.remove_special_chars(text)
        else:
            text=self.remove_special_chars(text)

        text=self.remove_extra_whitespace(text)

        return text.lower()
