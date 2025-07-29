"""
Model Training and Evaluation Module for Fake News Detection

This module provides comprehensive model training, evaluation, and comparison
capabilities for multiple machine learning algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A comprehensive class for training and evaluating multiple machine learning models
    for fake news detection.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Initialize models with default parameters
        self._initialize_models()
        
        print("ModelTrainer initialized with 3 models")
    
    def _initialize_models(self) -> None:
        """Initialize the machine learning models with default parameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
    
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: pd.Series, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split the data for training and testing.
        
        Args:
            X (np.ndarray): Feature matrix
            y (pd.Series): Target labels
            test_size (float): Proportion of data for testing
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        print(f"Preparing data with test_size={test_size}")
        
        # Encode labels if they are strings
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            y_encoded = y.values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, 
                   model_name: str, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        print(f"{model_name} training completed")
        
        return model
    
    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        print("Training all models...")
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        
        self.is_fitted = True
        print("All models trained successfully")
        
        return self.trained_models
    
    def evaluate_model(self, 
                      model_name: str, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a specific model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add AUC if probability predictions are available
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                metrics['auc_roc'] = None
        
        return metrics
    
    def evaluate_all_models(self, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            pd.DataFrame: Evaluation results for all models
        """
        print("Evaluating all models...")
        
        results = {}
        
        for model_name in self.trained_models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results[model_name] = metrics
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        self.evaluation_results = results_df
        return results_df
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            Tuple: (model_name, model_object, best_score)
        """
        if self.evaluation_results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all_models() first.")
        
        if metric not in self.evaluation_results.columns:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(self.evaluation_results.columns)}")
        
        best_model_name = self.evaluation_results[metric].idxmax()
        best_score = self.evaluation_results.loc[best_model_name, metric]
        best_model = self.trained_models[best_model_name]
        
        print(f"Best model: {best_model_name} with {metric} = {best_score:.4f}")
        
        return best_model_name, best_model, best_score
    
    def plot_confusion_matrix(self, 
                             model_name: str, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            save_path (Optional[str]): Path to save the plot
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['REAL', 'FAKE'], 
                   yticklabels=['REAL', 'FAKE'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def get_classification_report(self, 
                                 model_name: str, 
                                 X_test: np.ndarray, 
                                 y_test: np.ndarray) -> str:
        """
        Get detailed classification report for a specific model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            str: Classification report
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # Get class names
        if hasattr(self.label_encoder, 'classes_'):
            target_names = self.label_encoder.classes_
        else:
            target_names = ['REAL', 'FAKE']
        
        report = classification_report(y_test, y_pred, target_names=target_names)
        return report
    
    def perform_cross_validation(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                cv: int = 5) -> pd.DataFrame:
        """
        Perform cross-validation for all models.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            cv (int): Number of cross-validation folds
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            cv_results[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"{model_name} - Mean F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Convert to DataFrame
        cv_df = pd.DataFrame({
            name: [results['mean_score'], results['std_score']] 
            for name, results in cv_results.items()
        }, index=['mean_f1', 'std_f1']).T
        
        return cv_df
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison of model performance.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if self.evaluation_results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all_models() first.")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            if metric in self.evaluation_results.columns:
                self.evaluation_results[metric].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(self.evaluation_results[metric]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Example usage of the ModelTrainer class.
    """
    # This is a placeholder main function
    # In practice, this would be called from the main training script
    print("ModelTrainer module loaded successfully")
    print("Available models:", ['Logistic Regression', 'Naive Bayes', 'Random Forest'])


if __name__ == "__main__":
    main()
