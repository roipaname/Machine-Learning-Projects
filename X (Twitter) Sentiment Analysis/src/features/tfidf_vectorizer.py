"""
TF-IDF Feature Extraction for Text Classification.

This module implements TF-IDF vectorization with careful handling of:
- Vocabulary size control
- Document frequency filtering
- N-gram extraction
- Feature persistence
- Incremental updates
"""


from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pathlib
from typing import List,Dict,Optional,Union,Tuple
import numpy as np
from scipy.sparse import csr_matrix


from config.settings import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    SUBLINEAR_TF,
    USE_IDF,VECTORIZER_SAVE_PATH
    ,CUSTOM_STOPWORDS
)


class TFIDFVectorizer:
    """
    Wrapper around sklearn's TfidfVectorizer with project-specific configuration.
    
    This class provides:
    - Consistent feature extraction across training and inference
    - Model persistence
    - Feature analysis utilities
    - Vocabulary management
    """

    def __init__(self,
                 max_features:int=TFIDF_MAX_FEATURES,
                 min_df:Union[int,float]=TFIDF_MIN_DF,
                 max_df:Union[int,float]=TFIDF_MAX_DF,
                 ngram_range:Tuple[int,int]=TFIDF_NGRAM_RANGE,
                 use_idf:bool=USE_IDF,
                 sublinear_tf:bool=SUBLINEAR_TF,
                 custom_stopwords:Optional[List[str]]=None
                 ):
        self.max_features=max_features
        self.min_df=min_df
        self.max_df=max_df
        self.ngram_range=ngram_range
        self.use_idf=use_idf
        self.sublinear_tf=sublinear_tf
        self.stopwords=set(CUSTOM_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        self.vectorizer=self._create_vectorizer()

        self.isfitted=False
        self.vocabulary_size=0

    def _create_vectorizer(self):
        return TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words=list(self.stopwords) if self.stopwords else None,
            strip_accents='unicode',
            lowercase=True,  # Redundant if preprocessing done, but safe
            token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only words with 3+ letters
            dtype=np.float32,  # Memory efficiency
            norm='l2',  # L2 normalization for cosine similarity
            smooth_idf=True,  # Smooth IDF we
        )
    def fit(self, documents:List[str])->'TFIDFVectorizer':
        if not documents:
            logger.error("No documents submitted")
            raise
        try:
            self.vectorizer.fit(documents)
            self.isfitted=True
            self.vocabulary_size=len(self.vectorizer.vocabulary_)
            logger.succes(
                f"TF-IDF fitted successfully. Vocabulary size: {self.vocabulary_size}"
            )
        except Exception as e:
            logger.error(f"failed to fit vectorizer :{e}")
            raise
        return self
    def transform(self,documents: List[str])->csr_matrix:
        """
        Transform documents into TF-IDF feature matrix.
        
        Args:
            documents: List of preprocessed text documents
            
        Returns:
            Sparse matrix of shape (n_documents, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        if not documents:
            logger.warning("Transforming empty document list")
            return csr_matrix((0, self.vocabulary_size))
        
        logger.debug(f"Transforming {len(documents)} documents...")
        try:
            features=self.vectorizer.transform(documents)
            logger.debug(f"Generated feature matrix: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"Error transforming documents: {e}")
            raise
      