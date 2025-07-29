"""
Feature Extraction Module for Fake News Detection

This module provides TF-IDF vectorization and other feature extraction
techniques to convert text data into numerical vectors for machine learning.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional, Union
import joblib
import os


class FeatureExtractor:
    """
    A class for extracting numerical features from text data using various techniques.
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_idf: bool = True,
                 sublinear_tf: bool = True):
        """
        Initialize the FeatureExtractor.
        
        Args:
            max_features (int): Maximum number of features to extract
            min_df (int): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms (as ratio)
            ngram_range (Tuple[int, int]): Range of n-grams to extract
            use_idf (bool): Whether to use inverse document frequency
            sublinear_tf (bool): Whether to use sublinear term frequency scaling
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.svd_transformer = None
        
        # Feature names and statistics
        self.feature_names = None
        self.vocabulary_size = 0
        self.is_fitted = False
        
        print(f"FeatureExtractor initialized with max_features={max_features}")
    
    def _create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Create and configure TF-IDF vectorizer.
        
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        return TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words=None,  # We handle stopwords in preprocessing
            lowercase=False,  # We handle lowercasing in preprocessing
            token_pattern=r'\b\w+\b'  # Simple word pattern
        )
    
    def _create_count_vectorizer(self) -> CountVectorizer:
        """
        Create and configure Count vectorizer.
        
        Returns:
            CountVectorizer: Configured vectorizer
        """
        return CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=None,
            lowercase=False,
            token_pattern=r'\b\w+\b'
        )
    
    def fit_tfidf(self, texts: Union[pd.Series, list]) -> 'FeatureExtractor':
        """
        Fit the TF-IDF vectorizer on the training texts.
        
        Args:
            texts (Union[pd.Series, list]): Training texts
            
        Returns:
            FeatureExtractor: Self for method chaining
        """
        print("Fitting TF-IDF vectorizer...")
        
        # Create and fit vectorizer
        self.tfidf_vectorizer = self._create_tfidf_vectorizer()
        self.tfidf_vectorizer.fit(texts)
        
        # Store feature information
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        self.is_fitted = True
        
        print(f"TF-IDF vectorizer fitted with {self.vocabulary_size} features")
        return self
    
    def transform_tfidf(self, texts: Union[pd.Series, list]) -> np.ndarray:
        """
        Transform texts using the fitted TF-IDF vectorizer.
        
        Args:
            texts (Union[pd.Series, list]): Texts to transform
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if not self.is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        print(f"Transforming {len(texts)} texts using TF-IDF...")
        features = self.tfidf_vectorizer.transform(texts)
        
        print(f"Generated feature matrix with shape: {features.shape}")
        return features
    
    def fit_transform_tfidf(self, texts: Union[pd.Series, list]) -> np.ndarray:
        """
        Fit the TF-IDF vectorizer and transform texts in one step.
        
        Args:
            texts (Union[pd.Series, list]): Texts to fit and transform
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        return self.fit_tfidf(texts).transform_tfidf(texts)
    
    def fit_count_vectorizer(self, texts: Union[pd.Series, list]) -> 'FeatureExtractor':
        """
        Fit the Count vectorizer on the training texts.
        
        Args:
            texts (Union[pd.Series, list]): Training texts
            
        Returns:
            FeatureExtractor: Self for method chaining
        """
        print("Fitting Count vectorizer...")
        
        self.count_vectorizer = self._create_count_vectorizer()
        self.count_vectorizer.fit(texts)
        
        print(f"Count vectorizer fitted with {len(self.count_vectorizer.get_feature_names_out())} features")
        return self
    
    def transform_count(self, texts: Union[pd.Series, list]) -> np.ndarray:
        """
        Transform texts using the fitted Count vectorizer.
        
        Args:
            texts (Union[pd.Series, list]): Texts to transform
            
        Returns:
            np.ndarray: Count feature matrix
        """
        if self.count_vectorizer is None:
            raise ValueError("Count vectorizer not fitted. Call fit_count_vectorizer() first.")
        
        return self.count_vectorizer.transform(texts)
    
    def apply_dimensionality_reduction(self, 
                                     features: np.ndarray, 
                                     n_components: int = 1000,
                                     fit: bool = True) -> np.ndarray:
        """
        Apply dimensionality reduction using Truncated SVD.
        
        Args:
            features (np.ndarray): Input feature matrix
            n_components (int): Number of components to keep
            fit (bool): Whether to fit the transformer
            
        Returns:
            np.ndarray: Reduced feature matrix
        """
        if fit or self.svd_transformer is None:
            print(f"Applying SVD dimensionality reduction to {n_components} components...")
            self.svd_transformer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_features = self.svd_transformer.fit_transform(features)
        else:
            reduced_features = self.svd_transformer.transform(features)
        
        print(f"Reduced features from {features.shape[1]} to {reduced_features.shape[1]} dimensions")
        return reduced_features
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most important features based on model coefficients.
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Top features with their importance scores
        """
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Feature extractor not fitted or feature names not available")
        
        # Get feature importance scores
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute values of coefficients
            importance_scores = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
        
        # Create DataFrame with features and their importance
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance and return top N
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        return feature_importance_df.head(top_n)
    
    def get_vocabulary_stats(self) -> dict:
        """
        Get statistics about the extracted vocabulary.
        
        Returns:
            dict: Vocabulary statistics
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted")
        
        stats = {
            'vocabulary_size': self.vocabulary_size,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'use_idf': self.use_idf,
            'sublinear_tf': self.sublinear_tf
        }
        
        if self.tfidf_vectorizer is not None:
            # Get n-gram distribution
            unigrams = sum(1 for term in self.feature_names if len(term.split()) == 1)
            bigrams = sum(1 for term in self.feature_names if len(term.split()) == 2)
            
            stats.update({
                'unigrams': unigrams,
                'bigrams': bigrams,
                'unigram_ratio': unigrams / self.vocabulary_size,
                'bigram_ratio': bigrams / self.vocabulary_size
            })
        
        return stats
    
    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the vectorizer and metadata
        save_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'svd_transformer': self.svd_transformer,
            'feature_names': self.feature_names,
            'vocabulary_size': self.vocabulary_size,
            'is_fitted': self.is_fitted,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
                'use_idf': self.use_idf,
                'sublinear_tf': self.sublinear_tf
            }
        }
        
        joblib.dump(save_data, filepath)
        print(f"Feature extractor saved to {filepath}")
    
    def load_vectorizer(self, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath (str): Path to the saved vectorizer
            
        Returns:
            FeatureExtractor: Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        # Restore vectorizer and metadata
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        self.count_vectorizer = save_data['count_vectorizer']
        self.svd_transformer = save_data['svd_transformer']
        self.feature_names = save_data['feature_names']
        self.vocabulary_size = save_data['vocabulary_size']
        self.is_fitted = save_data['is_fitted']
        
        # Restore configuration
        config = save_data['config']
        self.max_features = config['max_features']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.ngram_range = config['ngram_range']
        self.use_idf = config['use_idf']
        self.sublinear_tf = config['sublinear_tf']
        
        print(f"Feature extractor loaded from {filepath}")
        return self


def main():
    """
    Example usage of the FeatureExtractor class.
    """
    # Sample texts
    texts = [
        "this is a sample news article about politics",
        "breaking news about technology and innovation",
        "sports news and latest updates from the field",
        "weather forecast and climate change discussion",
        "economic analysis and market trends report"
    ]
    
    # Initialize feature extractor
    extractor = FeatureExtractor(max_features=100, ngram_range=(1, 2))
    
    # Fit and transform
    features = extractor.fit_transform_tfidf(texts)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature matrix type: {type(features)}")
    
    # Get vocabulary statistics
    stats = extractor.get_vocabulary_stats()
    print("\nVocabulary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
