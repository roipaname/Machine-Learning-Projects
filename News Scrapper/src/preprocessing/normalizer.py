from typing import List
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
logging.basicConfig(level=logging.INFO)

# Download required NLTK data (run once)
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
        self.language=language
        self.stop_words=set(stopwords.words(language))
        self.lemmatizer=WordNetLemmatizer()

        # Custom stopwords for news articles
        self.custom_stopwords = {
            'said', 'told', 'says', 'according', 'report', 'reports',
            'article', 'news', 'source', 'sources'
        }
        self.stop_words.update(self.custom_stopwords)

    def tokenize(self,text:str)->List[str]:
        """Tokenize text"""
        try:
            tokens=word_tokenize(text)
            return tokens
        except Exception as e:
            logging.error(f"failed to tokenize {e}")
            return text.split()
        
    def remove_stopwords(self, tokens:List[str])->List[str]:
        """Filter out stopwords"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self,tokens:List[str])->List[str]:
        """"convert words to base form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_tokens(self,tokens:List[str],min_length:int=3,max_length:int=20)->List[str]:
        """remove very shprt/long tokens"""
        return [
            token for token in tokens if min_length <= len(token) <= max_length and token.isalpha()
        ]
    def normalize(self,text:str)->List[str]:
        """Full Normalization pipeline"""
        tokens=self.tokenize(text)
        tokens=self.remove_stopwords(tokens)
        tokens=self.filter_tokens(tokens)
        tokens=self.lemmatize(tokens)
        return tokens
    

        
if __name__ == "__main__":
    logging.info("Running TextNormalizer test...")

    sample_text = (
        "According to sources, the minister said the markets are crashing rapidly in 2024!"
    )

    normalizer = TextNormalizer(language="english")

    print("\nRaw text:")
    print(sample_text)

    tokens = normalizer.tokenize(sample_text)
    print("\nTokens:")
    print(tokens)

    tokens_no_stopwords = normalizer.remove_stopwords(tokens)
    print("\nAfter stopword removal:")
    print(tokens_no_stopwords)

    filtered_tokens = normalizer.filter_tokens(tokens_no_stopwords)
    print("\nAfter filtering:")
    print(filtered_tokens)

    lemmatized_tokens = normalizer.lemmatize(filtered_tokens)
    print("\nAfter lemmatization:")
    print(lemmatized_tokens)

    final_tokens = normalizer.normalize(sample_text)
    print("\nFinal normalized output:")
    print(final_tokens)

    logging.info("TextNormalizer test completed.")

