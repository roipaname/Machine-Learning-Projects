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