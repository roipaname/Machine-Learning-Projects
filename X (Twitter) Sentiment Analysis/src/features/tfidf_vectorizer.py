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
from pathlib import Path
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

        self.is_fitted=False
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

    def fit_transform(self, documents: List[str]) -> csr_matrix:
        """
        Fit vectorizer and transform documents in one step.
        
        Args:
            documents: List of preprocessed text documents
            
        Returns:
            Sparse matrix of shape (n_documents, n_features)
        """
        logger.info(f"Fitting and transforming {len(documents)} documents...")
        
        try:
            features = self.vectorizer.fit_transform(documents)
            self.is_fitted = True
            self.vocabulary_size = len(self.vectorizer.vocabulary_)
            
            logger.info(
                f"Fit-transform complete. Shape: {features.shape}, "
                f"Vocabulary: {self.vocabulary_size}"
            )
            
            self._log_feature_statistics()
            
            return features
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise
      
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names (vocabulary terms).
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")

        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get vocabulary mapping (term -> index).
        
        Returns:
            Dictionary mapping terms to feature indices
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.vocabulary_
    
    def get_idf_scores(self) -> Dict[str, float]:
        """
        Get IDF scores for all terms in vocabulary.
        
        Returns:
            Dictionary mapping terms to IDF scores
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        if not self.use_idf:
            logger.warning("IDF not used in this vectorizer")
            return {}
        feature_names=self.get_feature_names()
        idf_scores=self.vectorizer.idf_

        return Dict(zip(feature_names,idf_scores))
    def get_top_features_by_idf(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """
        Get terms with highest IDF scores (most distinctive).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (term, idf_score) tuples
        """
        idf_scores=self.get_idf_scores()
        sorted_scores=sorted(idf_scores.items(),key=lambda x:x[1],reverse=True)
        return sorted_scores[:top_n]
    
    def get_document_features(
        self, 
        document: str, 
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a single document.
        
        Args:
            document: Preprocessed text document
            top_n: Number of top features to return
            
        Returns:
            List of (term, tfidf_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        # Transform document
        tfidf_vector = self.transform([document])
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Get non-zero features
        nonzero_indices = tfidf_vector.nonzero()[1]
        nonzero_scores = tfidf_vector.data
        
        # Create list of (term, score) tuples
        features = [
            (feature_names[idx], score)
            for idx, score in zip(nonzero_indices, nonzero_scores)
        ]
        
        # Sort by score and return top N
        features.sort(key=lambda x: x[1], reverse=True)
        return features[:top_n]
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted vectorizer to disk.
        
        Args:
            filepath: Path to save file (uses default if None)
            
        Returns:
            Path where vectorizer was saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        save_path = filepath or VECTORIZER_SAVE_PATH
        save_path = Path(save_path)
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save entire extractor object (includes config)
            with open(save_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Vectorizer saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'TFIDFFeatureExtractor':
        """
        Load fitted vectorizer from disk.
        
        Args:
            filepath: Path to saved vectorizer (uses default if None)
            
        Returns:
            Loaded TFIDFFeatureExtractor instance
        """
        load_path = filepath or VECTORIZER_SAVE_PATH
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {load_path}")
        
        try:
            with open(load_path, 'rb') as f:
                extractor = pickle.load(f)
            
            if not isinstance(extractor, cls):
                raise TypeError(f"Loaded object is not {cls.__name__}")
            
            logger.info(
                f"Vectorizer loaded from {load_path}. "
                f"Vocabulary size: {extractor.vocabulary_size}"
            )
            
            return extractor
            
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            raise
    
    def _log_feature_statistics(self):
        """Log statistics about extracted features."""
        try:
            feature_names = self.get_feature_names()
            
            # Count n-grams
            unigrams = sum(1 for term in feature_names if ' ' not in term)
            bigrams = sum(1 for term in feature_names if term.count(' ') == 1)
            trigrams = sum(1 for term in feature_names if term.count(' ') == 2)
            
            logger.info(f"Feature statistics:")
            logger.info(f"  - Total features: {self.vocabulary_size}")
            logger.info(f"  - Unigrams: {unigrams}")
            logger.info(f"  - Bigrams: {bigrams}")
            logger.info(f"  - Trigrams: {trigrams}")
            
            # Log sample features
            sample_features = feature_names[:10]
            logger.debug(f"Sample features: {', '.join(sample_features)}")
            
            # Log top IDF terms if available
            if self.use_idf:
                top_idf = self.get_top_features_by_idf(top_n=10)
                top_terms = [term for term, _ in top_idf]
                logger.debug(f"Top IDF terms: {', '.join(top_terms)}")
                
        except Exception as e:
            logger.warning(f"Could not log feature statistics: {e}")
    
    def __repr__(self) -> str:
        """String representation of extractor."""
        status = "fitted" if self.is_fitted else "not fitted"
        vocab_info = f", vocab_size={self.vocabulary_size}" if self.is_fitted else ""
        return (
            f"TFIDFFeatureExtractor("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"status={status}{vocab_info})"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def save_vectorizer(
    extractor: TFIDFFeatureExtractor,
    filepath: Optional[Path] = None
) -> Path:
    """
    Convenience function to save vectorizer.
    
    Args:
        extractor: Fitted TFIDFFeatureExtractor instance
        filepath: Path to save file
        
    Returns:
        Path where vectorizer was saved
    """
    return extractor.save(filepath)


def load_vectorizer(filepath: Optional[Path] = None) -> TFIDFFeatureExtractor:
    """
    Convenience function to load vectorizer.
    
    Args:
        filepath: Path to saved vectorizer
        
    Returns:
        Loaded TFIDFFeatureExtractor instance
    """
    return TFIDFFeatureExtractor.load(filepath)


def extract_top_features(
    documents: List[str],
    labels: List[str],
    top_n: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract top TF-IDF features for each class.
    
    Useful for understanding what terms are most distinctive for each category.
    
    Args:
        documents: List of preprocessed text documents
        labels: Corresponding category labels
        top_n: Number of top features per class
        
    Returns:
        Dictionary mapping class labels to list of (term, avg_tfidf) tuples
    """
    if len(documents) != len(labels):
        raise ValueError("Documents and labels must have same length")
    
    logger.info(f"Extracting top {top_n} features per class...")
    
    # Fit vectorizer on all documents
    extractor = TFIDFFeatureExtractor()
    tfidf_matrix = extractor.fit_transform(documents)
    feature_names = extractor.get_feature_names()
    
    # Group documents by label
    unique_labels = sorted(set(labels))
    class_features = {}
    
    for label in unique_labels:
        # Get indices for this class
        class_indices = [i for i, l in enumerate(labels) if l == label]
        
        # Get TF-IDF vectors for this class
        class_vectors = tfidf_matrix[class_indices]
        
        # Calculate mean TF-IDF score per feature
        mean_tfidf = np.asarray(class_vectors.mean(axis=0)).flatten()
        
        # Get top features
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_features = [
            (feature_names[idx], mean_tfidf[idx])
            for idx in top_indices
        ]
        
        class_features[label] = top_features
        
        logger.debug(
            f"Class '{label}': Top term = '{top_features[0][0]}' "
            f"(score: {top_features[0][1]:.4f})"
        )
    
    logger.info(f"Extracted features for {len(unique_labels)} classes")
    return class_features


def analyze_feature_importance(
    documents: List[str],
    labels: List[str],
    output_file: Optional[Path] = None
) -> Dict:
    """
    Comprehensive feature importance analysis.
    
    Args:
        documents: List of preprocessed text documents
        labels: Corresponding category labels
        output_file: Optional path to save analysis results
        
    Returns:
        Dictionary containing feature analysis results
    """
    logger.info("Starting feature importance analysis...")
    
    # Extract top features per class
    class_features = extract_top_features(documents, labels, top_n=30)
    
    # Fit vectorizer
    extractor = TFIDFFeatureExtractor()
    extractor.fit(documents)
    
    # Get global IDF scores
    global_idf = extractor.get_top_features_by_idf(top_n=50)
    
    # Compile results
    results = {
        'class_features': class_features,
        'global_top_idf': global_idf,
        'vocabulary_size': extractor.vocabulary_size,
        'num_documents': len(documents),
        'num_classes': len(set(labels))
    }
    
    # Save if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Feature Importance Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Vocabulary Size: {results['vocabulary_size']}\n")
            f.write(f"Number of Documents: {results['num_documents']}\n")
            f.write(f"Number of Classes: {results['num_classes']}\n\n")
            
            f.write("Top Features by Class:\n")
            f.write("-" * 80 + "\n")
            for label, features in class_features.items():
                f.write(f"\n{label.upper()}:\n")
                for term, score in features[:10]:
                    f.write(f"  {term:30s} {score:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Global Top IDF Terms:\n")
            f.write("-" * 80 + "\n")
            for term, score in global_idf[:20]:
                f.write(f"  {term:30s} {score:.4f}\n")
        
        logger.info(f"Analysis saved to {output_file}")
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    """
    Example usage and testing of TF-IDF feature extraction.
    """
    
    
    # Sample documents (preprocessed)
    sample_docs = [
        "artificial intelligence machine learning deep neural network",
        "stock market investment trading financial economy",
        "football soccer world cup championship tournament",
        "python programming software development coding",
        "climate change global warming environment sustainability"
    ]
    
    sample_labels = [
        "technology",
        "business",
        "sports",
        "technology",
        "science"
    ]
    
    logger.info("Testing TF-IDF Feature Extractor...")
    
    # Initialize extractor
    extractor = TFIDFVectorizer(
        max_features=100,
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    features = extractor.fit_transform(sample_docs)
    logger.info(f"Feature matrix shape: {features.shape}")
    
    # Analyze single document
    doc_features = extractor.get_document_features(sample_docs[0], top_n=5)
    logger.info("Top features for first document:")
    for term, score in doc_features:
        logger.info(f"  {term}: {score:.4f}")
    
    # Extract class-specific features
    class_features = extract_top_features(sample_docs, sample_labels, top_n=3)
    logger.info("\nTop features per class:")
    for label, features in class_features.items():
        terms = [term for term, _ in features]
        logger.info(f"  {label}: {', '.join(terms)}")
    
    logger.success("TF-IDF testing complete!")