"""
FactForge - Main Training Script for News Authenticity Detection

This script orchestrates the complete machine learning pipeline:
1. Data loading and validation
2. Text preprocessing
3. Feature extraction
4. Model training and evaluation
5. Model selection and saving
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from predictor import FakeNewsPredictor


def main():
    """
    Main function to run the complete fake news detection pipeline.
    """
    print("=" * 60)
    print("FACTFORGE - AI NEWS AUTHENTICITY DETECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Load and validate data
        print("STEP 1: LOADING DATA")
        print("-" * 30)
        
        data_loader = DataLoader("data/news.csv")
        data = data_loader.load_data()
        data_loader.print_data_summary()
        
        # Validate data quality
        validation = data_loader.validate_data_quality()
        if not validation['is_valid']:
            print("WARNING: Data quality issues detected:")
            for check, result in validation.items():
                if check != 'is_valid' and result:
                    print(f"  - {check}: {result}")
            print()
        
        print("\n" + "=" * 60 + "\n")
        
        # Step 2: Text preprocessing
        print("STEP 2: TEXT PREPROCESSING")
        print("-" * 30)
        
        preprocessor = TextPreprocessor()
        processed_data = preprocessor.preprocess_dataframe(data, 'text', 'processed_text')
        
        # Show preprocessing example
        sample_idx = 0
        print(f"\nPreprocessing Example:")
        print(f"Original: {data.iloc[sample_idx]['text'][:100]}...")
        print(f"Processed: {processed_data.iloc[sample_idx]['processed_text'][:100]}...")
        
        # Get preprocessing statistics
        stats = preprocessor.get_preprocessing_stats(
            data.iloc[sample_idx]['text'], 
            processed_data.iloc[sample_idx]['processed_text']
        )
        print(f"Text reduction: {stats['reduction_ratio']:.1%}")
        print(f"Word reduction: {stats['word_reduction_ratio']:.1%}")
        
        print("\n" + "=" * 60 + "\n")
        
        # Step 3: Feature extraction
        print("STEP 3: FEATURE EXTRACTION")
        print("-" * 30)
        
        feature_extractor = FeatureExtractor(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        # Extract features
        X = feature_extractor.fit_transform_tfidf(processed_data['processed_text'])
        y = processed_data['label']
        
        # Show feature extraction statistics
        vocab_stats = feature_extractor.get_vocabulary_stats()
        print(f"Feature matrix shape: {X.shape}")
        print(f"Vocabulary size: {vocab_stats['vocabulary_size']}")
        print(f"Unigrams: {vocab_stats['unigrams']} ({vocab_stats['unigram_ratio']:.1%})")
        print(f"Bigrams: {vocab_stats['bigrams']} ({vocab_stats['bigram_ratio']:.1%})")
        
        print("\n" + "=" * 60 + "\n")
        
        # Step 4: Model training and evaluation
        print("STEP 4: MODEL TRAINING AND EVALUATION")
        print("-" * 30)
        
        model_trainer = ModelTrainer(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(X, y, test_size=0.2)
        
        # Train all models
        trained_models = model_trainer.train_all_models(X_train, y_train)
        
        # Evaluate all models
        print("\nEvaluating models...")
        evaluation_results = model_trainer.evaluate_all_models(X_test, y_test)
        
        print("\nMODEL PERFORMANCE COMPARISON:")
        print(evaluation_results)
        
        # Perform cross-validation
        print("\nPerforming cross-validation...")
        cv_results = model_trainer.perform_cross_validation(X, y, cv=5)
        print("\nCROSS-VALIDATION RESULTS:")
        print(cv_results)
        
        print("\n" + "=" * 60 + "\n")
        
        # Step 5: Model selection and saving
        print("STEP 5: MODEL SELECTION AND SAVING")
        print("-" * 30)
        
        # Get best model
        best_model_name, best_model, best_score = model_trainer.get_best_model('f1_score')
        
        # Get detailed classification report for best model
        print(f"\nDETAILED CLASSIFICATION REPORT - {best_model_name}:")
        print(model_trainer.get_classification_report(best_model_name, X_test, y_test))
        
        # Save the best model
        predictor = FakeNewsPredictor()
        
        # Get performance metrics for the best model
        best_metrics = evaluation_results.loc[best_model_name].to_dict()
        
        model_path = predictor.save_model(
            model=best_model,
            vectorizer=feature_extractor,
            preprocessor=preprocessor,
            label_encoder=model_trainer.label_encoder,
            model_name=best_model_name,
            performance_metrics=best_metrics
        )
        
        print(f"\nBest model saved to: {model_path}")
        
        print("\n" + "=" * 60 + "\n")
        
        # Step 6: Test the saved model
        print("STEP 6: TESTING SAVED MODEL")
        print("-" * 30)
        
        # Load the saved model
        test_predictor = FakeNewsPredictor()
        test_predictor.load_model(model_path)
        
        # Test with sample texts
        test_texts = [
            "Scientists have discovered a new planet made entirely of chocolate.",
            "The Federal Reserve announced interest rates will remain unchanged.",
            "Local man claims his pet goldfish can speak 17 languages.",
            "New study shows exercise reduces heart disease risk by 30%."
        ]
        
        print("Testing predictions on sample texts:")
        for i, text in enumerate(test_texts, 1):
            prediction = test_predictor.predict_news(text)
            if hasattr(best_model, 'predict_proba'):
                _, confidence = test_predictor.predict_with_confidence(text)
                print(f"{i}. {text[:50]}...")
                print(f"   Prediction: {prediction} (Confidence: {confidence:.3f})")
            else:
                print(f"{i}. {text[:50]}...")
                print(f"   Prediction: {prediction}")
            print()
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best Model: {best_model_name}")
        print(f"F1-Score: {best_score:.4f}")
        print(f"Model saved to: {model_path}")
        print("=" * 60)
        
        # Provide next steps
        print("\nNEXT STEPS:")
        print("1. Run 'python cli_interface.py' to test the model interactively")
        print("2. Run 'python web_app.py' to start the web interface")
        print("3. Use the saved model in your own applications")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Training failed. Please check the error message above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
