"""
Model Prediction and Persistence Module for Fake News Detection

This module provides functionality for saving/loading trained models and
making predictions on new text samples.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FakeNewsPredictor:
    """
    A class for making predictions using trained fake news detection models.
    Handles model persistence and provides easy-to-use prediction interface.
    """
    
    def __init__(self):
        """Initialize the FakeNewsPredictor."""
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.label_encoder = None
        self.model_info = {}
        self.is_loaded = False
        
    def save_model(self, 
                   model: Any,
                   vectorizer: Any,
                   preprocessor: Any,
                   label_encoder: Any,
                   model_name: str,
                   performance_metrics: Dict[str, float],
                   save_dir: str = "models/saved_models") -> str:
        """
        Save a trained model and its components.
        
        Args:
            model: Trained ML model
            vectorizer: Fitted feature extractor/vectorizer
            preprocessor: Text preprocessor
            label_encoder: Label encoder for target classes
            model_name (str): Name of the model
            performance_metrics (Dict[str, float]): Model performance metrics
            save_dir (str): Directory to save the model
            
        Returns:
            str: Path to the saved model file
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"factforge_model_{model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
        filepath = os.path.join(save_dir, filename)
        
        # Prepare model package
        model_package = {
            'model': model,
            'vectorizer': vectorizer,
            'preprocessor': preprocessor,
            'label_encoder': label_encoder,
            'model_info': {
                'model_name': model_name,
                'performance_metrics': performance_metrics,
                'save_timestamp': timestamp,
                'save_date': datetime.now().isoformat(),
                'model_type': type(model).__name__
            }
        }
        
        # Save the model package
        joblib.dump(model_package, filepath)
        
        print(f"Model saved successfully to: {filepath}")
        print(f"Model: {model_name}")
        print(f"Performance metrics: {performance_metrics}")
        
        return filepath
    
    def load_model(self, filepath: str) -> 'FakeNewsPredictor':
        """
        Load a saved model and its components.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            FakeNewsPredictor: Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Load the model package
            model_package = joblib.load(filepath)
            
            # Extract components
            self.model = model_package['model']
            self.vectorizer = model_package['vectorizer']
            self.preprocessor = model_package['preprocessor']
            self.label_encoder = model_package['label_encoder']
            self.model_info = model_package['model_info']
            
            self.is_loaded = True
            
            print(f"Model loaded successfully from: {filepath}")
            print(f"Model: {self.model_info['model_name']}")
            print(f"Saved on: {self.model_info['save_date']}")
            print(f"Performance: {self.model_info['performance_metrics']}")
            
            return self
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict_news(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict whether news text is FAKE or REAL.
        
        Args:
            text (Union[str, List[str]]): News text(s) to classify
            
        Returns:
            Union[str, List[str]]: Prediction(s) - 'FAKE' or 'REAL'
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Use load_model() first.")
        
        # Handle single text input
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        try:
            # Preprocess the text(s)
            processed_texts = []
            for t in texts:
                processed_text = self.preprocessor.preprocess_text(t)
                processed_texts.append(processed_text)
            
            # Vectorize the processed text(s)
            features = self.vectorizer.transform_tfidf(processed_texts)
            
            # Make predictions
            predictions = self.model.predict(features)
            
            # Convert predictions back to labels
            if hasattr(self.label_encoder, 'inverse_transform'):
                predicted_labels = self.label_encoder.inverse_transform(predictions)
            else:
                # Fallback if label encoder is not available
                predicted_labels = ['FAKE' if pred == 1 else 'REAL' for pred in predictions]
            
            # Return single prediction or list based on input
            if single_input:
                return predicted_labels[0]
            else:
                return list(predicted_labels)
                
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def predict_with_confidence(self, text: Union[str, List[str]]) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Predict with confidence scores.
        
        Args:
            text (Union[str, List[str]]): News text(s) to classify
            
        Returns:
            Union[Tuple[str, float], List[Tuple[str, float]]]: Prediction(s) with confidence
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Use load_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Loaded model doesn't support probability predictions")
        
        # Handle single text input
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        try:
            # Preprocess the text(s)
            processed_texts = []
            for t in texts:
                processed_text = self.preprocessor.preprocess_text(t)
                processed_texts.append(processed_text)
            
            # Vectorize the processed text(s)
            features = self.vectorizer.transform_tfidf(processed_texts)
            
            # Make predictions with probabilities
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            # Convert predictions back to labels and get confidence scores
            results = []
            for i, pred in enumerate(predictions):
                if hasattr(self.label_encoder, 'inverse_transform'):
                    label = self.label_encoder.inverse_transform([pred])[0]
                else:
                    label = 'FAKE' if pred == 1 else 'REAL'
                
                # Get confidence (max probability)
                confidence = np.max(probabilities[i])
                results.append((label, confidence))
            
            # Return single result or list based on input
            if single_input:
                return results[0]
            else:
                return results
                
        except Exception as e:
            raise ValueError(f"Error making prediction with confidence: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Use load_model() first.")
        
        return self.model_info.copy()
    
    def batch_predict(self, texts: List[str], batch_size: int = 100) -> List[str]:
        """
        Predict on a large batch of texts efficiently.
        
        Args:
            texts (List[str]): List of texts to classify
            batch_size (int): Size of each processing batch
            
        Returns:
            List[str]: List of predictions
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Use load_model() first.")
        
        all_predictions = []
        
        print(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.predict_news(batch_texts)
            
            if isinstance(batch_predictions, str):
                batch_predictions = [batch_predictions]
            
            all_predictions.extend(batch_predictions)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_texts)} texts...")
        
        print("Batch prediction completed")
        return all_predictions
    
    def evaluate_text_sample(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed evaluation of a single text sample.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Dict[str, Any]: Detailed evaluation results
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Use load_model() first.")
        
        # Get prediction with confidence
        if hasattr(self.model, 'predict_proba'):
            prediction, confidence = self.predict_with_confidence(text)
        else:
            prediction = self.predict_news(text)
            confidence = None
        
        # Get processed text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Get preprocessing statistics
        preprocessing_stats = self.preprocessor.get_preprocessing_stats(text, processed_text)
        
        # Prepare evaluation results
        evaluation = {
            'original_text': text,
            'processed_text': processed_text,
            'prediction': prediction,
            'confidence': confidence,
            'preprocessing_stats': preprocessing_stats,
            'model_info': {
                'model_name': self.model_info['model_name'],
                'model_type': self.model_info['model_type']
            }
        }
        
        return evaluation


def find_latest_model(models_dir: str = "models/saved_models") -> Optional[str]:
    """
    Find the most recently saved model in the models directory.
    
    Args:
        models_dir (str): Directory containing saved models
        
    Returns:
        Optional[str]: Path to the latest model file, or None if no models found
    """
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not model_files:
        return None
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    latest_model = os.path.join(models_dir, model_files[0])
    return latest_model


def main():
    """
    Example usage of the FakeNewsPredictor class.
    """
    # Example of how to use the predictor
    print("FakeNewsPredictor module loaded successfully")
    
    # Try to find and load the latest model
    latest_model_path = find_latest_model()
    
    if latest_model_path:
        print(f"Latest model found: {latest_model_path}")
        
        # Example of loading and using the model
        # predictor = FakeNewsPredictor()
        # predictor.load_model(latest_model_path)
        # 
        # sample_text = "Breaking news: Scientists discover chocolate planet!"
        # prediction = predictor.predict_news(sample_text)
        # print(f"Prediction: {prediction}")
    else:
        print("No saved models found. Train a model first using main.py")


if __name__ == "__main__":
    main()
