from typing import List
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import logging
logging.basicConfig(level=logging.INFO)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class TextNormalizer:
    """Tokenize, rmeove stop words,and lemmatize"""
    def __init__(self,language:str='english'):
        self.langauage=language
        self.stop_words=set(stopwords.words(language))
        self.lemmatizer=word_tokenize()

        self.custom_stopwords={
            "im", "ive", "id", "dont", "cant", "wont",
            "u", "ur", "ya", "lol", "lmao", "rofl",
            "omg", "smh", "nah", "yeah", "yep",
            "pls", "plz", "tho", "tho", "bc"
        }
        self.stop_words.update(self.custom_stopwords)

    def tokenize(self,text:str)->List[str]:
        try:
            tokens=word_tokenize(text)
            return tokens
        except Exception as e:
            logging.error("failed to tokenize {e}, defaulting to spliting")
            return text.split()
    def remove_stopwords(self,tokens:List[str])->List[str]:
        """Filter out stopwords"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    def lemmatize(self,tokens:List[str])->List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_tokens(self,tokens:List[str],min_length:int=3,max_length:int=25)->List[str]:
        return [
            token for token in tokens  if min_length <= len(token)<= max_length and token.isalpha()
        ]
    def normalize(self,text:str)->List[str]:
        """Full Normalization pipeline"""
        tokens=self.tokenize(text)
        tokens=self.remove_stopwords(tokens)
        tokens=self.filter_tokens(tokens)
        tokens=self.lemmatize(tokens)
        return tokens


        