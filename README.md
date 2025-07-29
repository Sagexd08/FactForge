# Fake News Detection Model

A comprehensive machine learning project for detecting fake news using Natural Language Processing and various classification algorithms.

## Features

- **Text Preprocessing**: Advanced text cleaning with NLTK
- **Multiple ML Models**: Logistic Regression, Naive Bayes, Random Forest
- **Feature Extraction**: TF-IDF Vectorization
- **Model Evaluation**: Comprehensive metrics and comparison
- **Prediction Interface**: CLI and Web interface options
- **Model Persistence**: Save and load trained models

## Installation

### 1. Clone or Download the Project
```bash
cd "Fake news detection"
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Download SpaCy Model (Optional)
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
Fake news detection/
├── data/
│   └── news.csv                 # Sample dataset
├── models/
│   └── saved_models/           # Trained models
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessor.py         # Text preprocessing
│   ├── feature_extractor.py    # TF-IDF vectorization
│   ├── model_trainer.py        # Model training and evaluation
│   └── predictor.py            # Prediction utilities
├── main.py                     # Main training script
├── cli_interface.py            # Command line interface
├── web_app.py                  # Flask web application
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Usage

### 1. Train the Model
```bash
python main.py
```

### 2. Use Command Line Interface
```bash
python cli_interface.py
```

### 3. Run Web Application
```bash
python web_app.py
```

## Dataset Format

The `news.csv` file should have two columns:
- `text`: The news article content
- `label`: Either "FAKE" or "REAL"

## Model Performance

The project trains and compares three models:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Contributing

Feel free to contribute by:
- Adding new preprocessing techniques
- Implementing additional ML models
- Improving the web interface
- Adding more evaluation metrics

## License

This project is for educational purposes.
