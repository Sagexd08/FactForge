"""
Text Preprocessing Module for Fake News Detection

This module provides comprehensive text preprocessing capabilities including:
- Text cleaning and normalization
- Stopword removal
- Tokenization and lemmatization
- Punctuation and digit removal
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Union
import pandas as pd
from tqdm import tqdm


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for cleaning and preparing text data
    for machine learning models.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the TextPreprocessor.
        
        Args:
            language (str): Language for stopwords (default: 'english')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words(self.language))
        except LookupError:
            print(f"Stopwords for {self.language} not found. Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(self.language))
        
        # Add custom stopwords that might be common in news
        custom_stopwords = {
            'said', 'says', 'according', 'report', 'reports', 'news', 'article',
            'story', 'breaking', 'update', 'latest', 'today', 'yesterday',
            'tomorrow', 'week', 'month', 'year', 'time', 'times'
        }
        self.stop_words.update(custom_stopwords)
        
        print(f"TextPreprocessor initialized with {len(self.stop_words)} stopwords")
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages."""
        required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{package}')
                except LookupError:
                    print(f"Downloading NLTK package: {package}")
                    nltk.download(package, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_punctuation_and_digits(self, text: str) -> str:
        """
        Remove punctuation and digits from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without punctuation and digits
        """
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace created by removals
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            # Fallback to simple split
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_short_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """
        Filter out tokens that are too short.
        
        Args:
            tokens (List[str]): List of tokens
            min_length (int): Minimum token length
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if len(token) >= min_length]
    
    def preprocess_text(self, text: str, return_string: bool = True) -> Union[str, List[str]]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Input text
            return_string (bool): If True, return joined string; if False, return token list
            
        Returns:
            Union[str, List[str]]: Preprocessed text or tokens
        """
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Remove punctuation and digits
        text = self.remove_punctuation_and_digits(text)
        
        # Step 3: Tokenization
        tokens = self.tokenize_text(text)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 5: Lemmatization
        tokens = self.lemmatize_tokens(tokens)
        
        # Step 6: Filter short tokens
        tokens = self.filter_short_tokens(tokens)
        
        if return_string:
            return ' '.join(tokens)
        else:
            return tokens
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                           new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess text data in a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text
            new_column (str): Name of the new column for processed text
            
        Returns:
            pd.DataFrame: DataFrame with processed text column
        """
        df_copy = df.copy()
        
        print(f"Preprocessing {len(df_copy)} text samples...")
        
        # Apply preprocessing with progress bar
        tqdm.pandas(desc="Processing text")
        df_copy[new_column] = df_copy[text_column].progress_apply(self.preprocess_text)
        
        # Remove rows where processed text is empty
        initial_length = len(df_copy)
        df_copy = df_copy[df_copy[new_column].str.strip() != '']
        final_length = len(df_copy)
        
        if initial_length != final_length:
            print(f"Removed {initial_length - final_length} rows with empty processed text")
        
        print(f"Preprocessing complete. Final dataset size: {final_length}")
        
        return df_copy
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> dict:
        """
        Get statistics about the preprocessing transformation.
        
        Args:
            original_text (str): Original text
            processed_text (str): Processed text
            
        Returns:
            dict: Preprocessing statistics
        """
        original_tokens = original_text.split()
        processed_tokens = processed_text.split()
        
        stats = {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'original_word_count': len(original_tokens),
            'processed_word_count': len(processed_tokens),
            'reduction_ratio': 1 - (len(processed_text) / len(original_text)) if len(original_text) > 0 else 0,
            'word_reduction_ratio': 1 - (len(processed_tokens) / len(original_tokens)) if len(original_tokens) > 0 else 0
        }
        
        return stats


def main():
    """
    Example usage of the TextPreprocessor class.
    """
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example text
    sample_text = """
    Breaking News: Scientists have discovered a new planet that is completely made of chocolate! 
    This amazing discovery was announced today at 3:30 PM. Dr. John Smith said, "This is incredible!"
    Visit our website at www.example.com for more details. Email us at news@example.com.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Preprocess the text
    processed_text = preprocessor.preprocess_text(sample_text)
    
    print("Processed Text:")
    print(processed_text)
    print("\n" + "="*50 + "\n")
    
    # Get preprocessing statistics
    stats = preprocessor.get_preprocessing_stats(sample_text, processed_text)
    
    print("Preprocessing Statistics:")
    for key, value in stats.items():
        if 'ratio' in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
