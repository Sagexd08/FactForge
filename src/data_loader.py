"""
Data Loading Module for Fake News Detection

This module handles loading and basic validation of the news dataset.
It provides utilities to load CSV data and perform initial data exploration.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os


class DataLoader:
    """
    A class to handle data loading and basic validation for the fake news detection project.
    """
    
    def __init__(self, data_path: str = "data/news.csv"):
        """
        Initialize the DataLoader with the path to the dataset.
        
        Args:
            data_path (str): Path to the CSV file containing the news data
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the news dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset with 'text' and 'label' columns
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            # Check if file exists
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at: {self.data_path}")
            
            # Load the CSV file
            self.data = pd.read_csv(self.data_path)
            
            # Validate required columns
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with missing values
            initial_shape = self.data.shape
            self.data = self.data.dropna()
            
            if self.data.shape[0] < initial_shape[0]:
                print(f"Removed {initial_shape[0] - self.data.shape[0]} rows with missing values")
            
            print(f"Successfully loaded {len(self.data)} news articles")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the loaded dataset.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'total_articles': len(self.data),
            'fake_articles': len(self.data[self.data['label'] == 'FAKE']),
            'real_articles': len(self.data[self.data['label'] == 'REAL']),
            'unique_labels': self.data['label'].unique().tolist(),
            'avg_text_length': self.data['text'].str.len().mean(),
            'min_text_length': self.data['text'].str.len().min(),
            'max_text_length': self.data['text'].str.len().max()
        }
        
        # Calculate class distribution
        info['class_distribution'] = self.data['label'].value_counts().to_dict()
        info['class_balance_ratio'] = min(info['class_distribution'].values()) / max(info['class_distribution'].values())
        
        return info
    
    def print_data_summary(self) -> None:
        """
        Print a comprehensive summary of the dataset.
        """
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return
        
        info = self.get_data_info()
        
        print("=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Articles: {info['total_articles']}")
        print(f"Fake Articles: {info['fake_articles']}")
        print(f"Real Articles: {info['real_articles']}")
        print(f"Unique Labels: {info['unique_labels']}")
        print(f"Class Balance Ratio: {info['class_balance_ratio']:.2f}")
        print()
        print("TEXT LENGTH STATISTICS:")
        print(f"Average Length: {info['avg_text_length']:.1f} characters")
        print(f"Minimum Length: {info['min_text_length']} characters")
        print(f"Maximum Length: {info['max_text_length']} characters")
        print()
        print("CLASS DISTRIBUTION:")
        for label, count in info['class_distribution'].items():
            percentage = (count / info['total_articles']) * 100
            print(f"{label}: {count} articles ({percentage:.1f}%)")
        print("=" * 50)
    
    def get_sample_articles(self, n: int = 3) -> pd.DataFrame:
        """
        Get sample articles from each class for inspection.
        
        Args:
            n (int): Number of samples per class
            
        Returns:
            pd.DataFrame: Sample articles
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        samples = []
        for label in self.data['label'].unique():
            label_data = self.data[self.data['label'] == label].sample(n=min(n, len(self.data[self.data['label'] == label])))
            samples.append(label_data)
        
        return pd.concat(samples, ignore_index=True)
    
    def validate_data_quality(self) -> dict:
        """
        Perform data quality checks.
        
        Returns:
            dict: Dictionary containing validation results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        validation_results = {
            'has_missing_values': self.data.isnull().any().any(),
            'has_empty_text': (self.data['text'].str.strip() == '').any(),
            'has_valid_labels': set(self.data['label'].unique()).issubset({'FAKE', 'REAL'}),
            'min_text_length_ok': self.data['text'].str.len().min() > 10,  # At least 10 characters
            'has_duplicates': self.data.duplicated().any()
        }
        
        validation_results['is_valid'] = all([
            not validation_results['has_missing_values'],
            not validation_results['has_empty_text'],
            validation_results['has_valid_labels'],
            validation_results['min_text_length_ok'],
            not validation_results['has_duplicates']
        ])
        
        return validation_results


def main():
    """
    Example usage of the DataLoader class.
    """
    # Initialize data loader
    loader = DataLoader()
    
    try:
        # Load the data
        data = loader.load_data()
        
        # Print summary
        loader.print_data_summary()
        
        # Show sample articles
        print("\nSAMPLE ARTICLES:")
        print("-" * 50)
        samples = loader.get_sample_articles(n=2)
        for idx, row in samples.iterrows():
            print(f"Label: {row['label']}")
            print(f"Text: {row['text'][:100]}...")
            print("-" * 50)
        
        # Validate data quality
        validation = loader.validate_data_quality()
        print(f"\nData Quality Check: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        for check, result in validation.items():
            if check != 'is_valid':
                print(f"  {check}: {result}")
                
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
