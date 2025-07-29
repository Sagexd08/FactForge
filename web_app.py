"""
Flask Web Application for Fake News Detection

This script provides a simple web interface for the fake news detection model.
Users can input news text and get real-time predictions through a web browser.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, flash, session
from datetime import datetime
import json
import uuid

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predictor import FakeNewsPredictor, find_latest_model


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'fake_news_detection_secret_key_2024'

# Global predictor instance
predictor = None
model_info = None
prediction_history = []


def load_model():
    """Load the trained model at startup."""
    global predictor, model_info
    
    try:
        # Find the latest model
        model_path = find_latest_model()
        if model_path is None:
            print("❌ No trained models found!")
            print("Please run 'python main.py' to train a model first.")
            return False
        
        # Load the model
        predictor = FakeNewsPredictor()
        predictor.load_model(model_path)
        model_info = predictor.get_model_info()
        
        print(f"✅ Model loaded: {model_info['model_name']}")
        print(f"📈 F1-Score: {model_info['performance_metrics'].get('f1_score', 'N/A'):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html', model_info=model_info)

@app.route('/analyze')
def analyze():
    """Analysis page with the prediction form."""
    return render_template('analyze.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    global prediction_history
    try:
        # Get text from form
        text = request.form.get('text', '').strip()

        if not text:
            return jsonify({
                'error': 'Please enter some text to analyze'
            }), 400

        if len(text) < 10:
            return jsonify({
                'error': 'Text is too short. Please enter at least 10 characters.'
            }), 400

        # Make prediction
        if hasattr(predictor.model, 'predict_proba'):
            prediction, confidence = predictor.predict_with_confidence(text)
        else:
            prediction = predictor.predict_news(text)
            confidence = None

        # Get detailed evaluation
        evaluation = predictor.evaluate_text_sample(text)

        # Create history entry
        history_entry = {
            'id': str(uuid.uuid4()),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'full_text': text,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'word_count': len(text.split()),
            'processed_word_count': len(evaluation['processed_text'].split())
        }

        # Add to history (keep last 50 entries)
        prediction_history.insert(0, history_entry)
        if len(prediction_history) > 50:
            prediction_history.pop()

        # Prepare response
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'text_length': len(text),
            'processed_text_length': len(evaluation['processed_text']),
            'word_count': len(text.split()),
            'processed_word_count': len(evaluation['processed_text'].split()),
            'reduction_ratio': evaluation['preprocessing_stats']['reduction_ratio'],
            'timestamp': datetime.now().isoformat(),
            'history_id': history_entry['id']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/model-info')
def api_model_info():
    """API endpoint to get model information."""
    if model_info:
        return jsonify(model_info)
    else:
        return jsonify({'error': 'No model loaded'}), 500


@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Handle batch predictions."""
    if request.method == 'GET':
        return render_template('batch.html', model_info=model_info)
    
    try:
        # Get texts from form (one per line)
        texts_input = request.form.get('texts', '').strip()
        
        if not texts_input:
            flash('Please enter some texts to analyze', 'error')
            return render_template('batch.html', model_info=model_info)
        
        # Split into individual texts
        texts = [line.strip() for line in texts_input.split('\n') if line.strip()]
        
        if len(texts) > 50:
            flash('Maximum 50 texts allowed for batch processing', 'error')
            return render_template('batch.html', model_info=model_info)
        
        # Make predictions
        predictions = predictor.batch_predict(texts)
        
        # Prepare results
        results = []
        fake_count = 0
        real_count = 0
        
        for i, (text, prediction) in enumerate(zip(texts, predictions), 1):
            if prediction == 'FAKE':
                fake_count += 1
            else:
                real_count += 1
            
            results.append({
                'index': i,
                'text': text,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'prediction': prediction
            })
        
        summary = {
            'total': len(texts),
            'fake': fake_count,
            'real': real_count,
            'fake_percentage': (fake_count / len(texts)) * 100,
            'real_percentage': (real_count / len(texts)) * 100
        }
        
        return render_template('batch_results.html', 
                             results=results, 
                             summary=summary,
                             model_info=model_info)
        
    except Exception as e:
        flash(f'Batch prediction error: {str(e)}', 'error')
        return render_template('batch.html', model_info=model_info)


@app.route('/history')
def history():
    """History page showing recent predictions."""
    return render_template('history.html',
                         history=prediction_history,
                         model_info=model_info)

@app.route('/statistics')
def statistics():
    """Statistics dashboard."""
    # Calculate statistics from history
    total_predictions = len(prediction_history)
    fake_count = sum(1 for h in prediction_history if h['prediction'] == 'FAKE')
    real_count = total_predictions - fake_count

    avg_confidence = 0
    if prediction_history:
        confidences = [h['confidence'] for h in prediction_history if h['confidence'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    stats = {
        'total_predictions': total_predictions,
        'fake_count': fake_count,
        'real_count': real_count,
        'fake_percentage': (fake_count / total_predictions * 100) if total_predictions > 0 else 0,
        'real_percentage': (real_count / total_predictions * 100) if total_predictions > 0 else 0,
        'avg_confidence': avg_confidence,
        'recent_predictions': prediction_history[:10]  # Last 10 predictions
    }

    return render_template('statistics.html', stats=stats, model_info=model_info)

@app.route('/model-details')
def model_details():
    """Detailed model information page."""
    return render_template('model_details.html', model_info=model_info)

@app.route('/about')
def about():
    """About page with model and project information."""
    return render_template('about.html', model_info=model_info)

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history."""
    global prediction_history
    prediction_history = []
    return jsonify({'success': True, 'message': 'History cleared successfully'})

@app.route('/api/export-history')
def export_history():
    """Export prediction history as JSON."""
    return jsonify({
        'history': prediction_history,
        'exported_at': datetime.now().isoformat(),
        'total_entries': len(prediction_history)
    })


# Create templates directory and HTML files
def create_templates():
    """Create HTML templates for the web application."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Base template with modern design
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}FactForge - AI News Authenticity Detector{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --success-color: #059669;
            --danger-color: #dc2626;
            --warning-color: #d97706;
            --info-color: #0891b2;
            --dark-color: #1f2937;
            --light-color: #f8fafc;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .navbar {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }

        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--primary-color) !important;
        }

        .nav-link {
            color: var(--dark-color) !important;
            font-weight: 500;
            margin: 0 10px;
            transition: all 0.3s ease;
        }

        .nav-link:hover, .nav-link.active {
            color: var(--primary-color) !important;
            transform: translateY(-2px);
        }

        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 20px 0;
            padding: 30px;
        }

        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }

        .btn {
            border-radius: 10px;
            font-weight: 600;
            padding: 12px 30px;
            transition: all 0.3s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }

        .prediction-result {
            margin-top: 20px;
            padding: 25px;
            border-radius: 15px;
            animation: slideIn 0.5s ease;
        }

        .fake-result {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #f87171;
        }

        .real-result {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border: 2px solid #4ade80;
        }

        .confidence-bar {
            height: 25px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }

        .confidence-fill {
            height: 100%;
            transition: width 1s ease;
            border-radius: 15px;
        }

        .stats-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 5px solid var(--primary-color);
        }

        .history-item {
            background: #f8fafc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #e5e7eb;
            transition: all 0.3s ease;
        }

        .history-item:hover {
            background: #f1f5f9;
            border-left-color: var(--primary-color);
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 20px;
            color: var(--primary-color);
        }

        .footer {
            background: rgba(31, 41, 55, 0.9);
            color: white;
            padding: 40px 0;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-hammer me-2"></i>FactForge
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/analyze"><i class="fas fa-search me-1"></i>Analyze</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/batch"><i class="fas fa-layer-group me-1"></i>Batch</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history"><i class="fas fa-history me-1"></i>History</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/statistics"><i class="fas fa-chart-bar me-1"></i>Statistics</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/model-details"><i class="fas fa-cog me-1"></i>Model</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/about"><i class="fas fa-info-circle me-1"></i>About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div style="padding-top: 80px;">
        <div class="container">
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ 'danger' if category == 'error' else 'info' }} alert-dismissible fade show">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}

            <div class="main-container">
                {% block content %}{% endblock %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(base_template)

    # Dashboard template
    dashboard_template = '''{% extends "base.html" %}

{% block title %}Dashboard - FactForge{% endblock %}

{% block content %}
<div class="text-center mb-5">
    <h1 class="display-4 fw-bold text-primary mb-3">
        <i class="fas fa-hammer me-3"></i>FactForge
    </h1>
    <p class="lead text-muted">Advanced AI system for forging truth from news - detecting authenticity with precision</p>

    {% if model_info %}
    <div class="alert alert-info d-inline-block">
        <i class="fas fa-robot me-2"></i>
        <strong>{{ model_info.model_name }}</strong> |
        Accuracy: <strong>{{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) }}%</strong> |
        F1-Score: <strong>{{ "%.3f"|format(model_info.performance_metrics.f1_score) }}</strong>
    </div>
    {% endif %}
</div>

<div class="row g-4 mb-5">
    <div class="col-md-4">
        <div class="card h-100 text-center">
            <div class="card-body">
                <div class="feature-icon">
                    <i class="fas fa-search-plus"></i>
                </div>
                <h5 class="card-title">Quick Analysis</h5>
                <p class="card-text">Analyze individual news articles for authenticity with confidence scores</p>
                <a href="/analyze" class="btn btn-primary">
                    <i class="fas fa-arrow-right me-2"></i>Start Analysis
                </a>
            </div>
        </div>
    </div>

    <div class="col-md-4">
        <div class="card h-100 text-center">
            <div class="card-body">
                <div class="feature-icon">
                    <i class="fas fa-layer-group"></i>
                </div>
                <h5 class="card-title">Batch Processing</h5>
                <p class="card-text">Process multiple news articles simultaneously for bulk analysis</p>
                <a href="/batch" class="btn btn-success">
                    <i class="fas fa-upload me-2"></i>Batch Analyze
                </a>
            </div>
        </div>
    </div>

    <div class="col-md-4">
        <div class="card h-100 text-center">
            <div class="card-body">
                <div class="feature-icon">
                    <i class="fas fa-history"></i>
                </div>
                <h5 class="card-title">Analysis History</h5>
                <p class="card-text">View your previous analyses and track prediction patterns</p>
                <a href="/history" class="btn btn-info">
                    <i class="fas fa-clock me-2"></i>View History
                </a>
            </div>
        </div>
    </div>
</div>

<div class="row g-4 mb-5">
    <div class="col-md-6">
        <div class="card stats-card">
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-chart-line me-2"></i>Performance Statistics
                </h5>
                <div class="row text-center">
                    <div class="col-6">
                        <h3 class="text-primary">{{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) if model_info else "N/A" }}%</h3>
                        <small class="text-muted">Accuracy</small>
                    </div>
                    <div class="col-6">
                        <h3 class="text-success">{{ "%.3f"|format(model_info.performance_metrics.f1_score) if model_info else "N/A" }}</h3>
                        <small class="text-muted">F1-Score</small>
                    </div>
                </div>
                <a href="/statistics" class="btn btn-outline-primary btn-sm mt-3">
                    <i class="fas fa-chart-bar me-1"></i>Detailed Stats
                </a>
            </div>
        </div>
    </div>

    <div class="col-md-6">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-cogs me-2"></i>Model Information
                </h5>
                {% if model_info %}
                <p><strong>Model:</strong> {{ model_info.model_name }}</p>
                <p><strong>Type:</strong> {{ model_info.model_type }}</p>
                <p><strong>Trained:</strong> {{ model_info.save_date[:10] }}</p>
                {% else %}
                <p class="text-muted">No model information available</p>
                {% endif %}
                <a href="/model-details" class="btn btn-outline-secondary btn-sm">
                    <i class="fas fa-info-circle me-1"></i>Full Details
                </a>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">
                    <i class="fas fa-lightbulb me-2"></i>How It Works
                </h5>
                <div class="row">
                    <div class="col-md-3 text-center mb-3">
                        <div class="feature-icon text-primary" style="font-size: 2rem;">
                            <i class="fas fa-file-text"></i>
                        </div>
                        <h6>1. Input Text</h6>
                        <small class="text-muted">Enter news article text</small>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="feature-icon text-success" style="font-size: 2rem;">
                            <i class="fas fa-cog"></i>
                        </div>
                        <h6>2. Preprocessing</h6>
                        <small class="text-muted">Clean and tokenize text</small>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="feature-icon text-warning" style="font-size: 2rem;">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h6>3. AI Analysis</h6>
                        <small class="text-muted">ML model prediction</small>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="feature-icon text-info" style="font-size: 2rem;">
                            <i class="fas fa-check-circle"></i>
                        </div>
                        <h6>4. Result</h6>
                        <small class="text-muted">FAKE or REAL with confidence</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(dashboard_template)

    # Enhanced analyze template
    analyze_template = '''{% extends "base.html" %}

{% block title %}Analyze News - FactForge{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="text-center mb-4">
            <h1 class="display-5 fw-bold text-primary">
                <i class="fas fa-search me-3"></i>FactForge Analysis
            </h1>
            <p class="lead text-muted">Enter news text below to forge the truth - AI-powered authenticity detection</p>
        </div>

        {% if model_info %}
        <div class="alert alert-info text-center">
            <i class="fas fa-robot me-2"></i>
            <strong>Active Model:</strong> {{ model_info.model_name }} |
            <strong>Accuracy:</strong> {{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) }}% |
            <strong>F1-Score:</strong> {{ "%.3f"|format(model_info.performance_metrics.f1_score) }}
        </div>
        {% endif %}

        <div class="card">
            <div class="card-body">
                <form id="predictionForm">
                    <div class="mb-4">
                        <label for="text" class="form-label fw-bold">
                            <i class="fas fa-newspaper me-2"></i>News Article Text
                        </label>
                        <textarea class="form-control" id="text" name="text" rows="8"
                                 placeholder="Paste your news article text here for analysis..."
                                 required style="border-radius: 10px; border: 2px solid #e5e7eb;"></textarea>
                        <div class="form-text">
                            <i class="fas fa-info-circle me-1"></i>
                            Minimum 10 characters required. Longer texts provide better analysis.
                        </div>
                        <div class="mt-2">
                            <small class="text-muted">
                                Characters: <span id="charCount">0</span> |
                                Words: <span id="wordCount">0</span>
                            </small>
                        </div>
                    </div>

                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <span id="submitText">
                                <i class="fas fa-search me-2"></i>Analyze News Article
                            </span>
                            <span id="loadingSpinner" class="spinner-border spinner-border-sm ms-2" style="display: none;"></span>
                        </button>
                    </div>
                </form>

                <div id="result" style="display: none;"></div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <i class="fas fa-shield-alt text-success" style="font-size: 2rem;"></i>
                        <h6 class="mt-2">Real News Example</h6>
                        <small class="text-muted">Factual, verifiable information from reliable sources</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <i class="fas fa-exclamation-triangle text-danger" style="font-size: 2rem;"></i>
                        <h6 class="mt-2">Fake News Example</h6>
                        <small class="text-muted">Misleading, false, or unverifiable claims</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Character and word counter
document.getElementById('text').addEventListener('input', function() {
    const text = this.value;
    document.getElementById('charCount').textContent = text.length;
    document.getElementById('wordCount').textContent = text.trim().split(/\\s+/).filter(word => word.length > 0).length;
});

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultDiv = document.getElementById('result');

    // Show loading state
    submitBtn.disabled = true;
    submitText.innerHTML = '<i class="fas fa-cog fa-spin me-2"></i>Analyzing...';
    loadingSpinner.style.display = 'inline-block';
    resultDiv.style.display = 'none';

    try {
        const formData = new FormData(e.target);
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data);
        } else {
            displayError(data.error);
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitText.innerHTML = '<i class="fas fa-search me-2"></i>Analyze News Article';
        loadingSpinner.style.display = 'none';
    }
});

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const isFake = data.prediction === 'FAKE';
    const confidence = data.confidence || 0.5;

    const confidenceColor = confidence > 0.8 ? 'success' : confidence > 0.6 ? 'warning' : 'danger';
    const predictionIcon = isFake ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const predictionColor = isFake ? 'danger' : 'success';

    resultDiv.innerHTML = `
        <div class="prediction-result ${isFake ? 'fake-result' : 'real-result'} mt-4">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <h3 class="mb-3">
                        <i class="${predictionIcon} text-${predictionColor} me-2"></i>
                        ${data.prediction}
                    </h3>
                    ${data.confidence ? `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-1">
                                <span><strong>Confidence Score</strong></span>
                                <span><strong>${(confidence * 100).toFixed(1)}%</strong></span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill bg-${predictionColor}"
                                     style="width: ${confidence * 100}%"></div>
                            </div>
                            <small class="text-muted mt-1">
                                Certainty Level: ${confidence > 0.8 ? 'Very High' : confidence > 0.6 ? 'High' : confidence > 0.5 ? 'Moderate' : 'Low'}
                            </small>
                        </div>
                    ` : ''}
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6><i class="fas fa-chart-bar me-2"></i>Analysis Details</h6>
                            <small class="text-muted">
                                <strong>Original:</strong> ${data.word_count} words<br>
                                <strong>Processed:</strong> ${data.processed_word_count} words<br>
                                <strong>Reduction:</strong> ${(data.reduction_ratio * 100).toFixed(1)}%<br>
                                <strong>Analyzed:</strong> ${new Date(data.timestamp).toLocaleString()}
                            </small>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-3 text-center">
                <button class="btn btn-outline-primary btn-sm me-2" onclick="window.location.href='/history'">
                    <i class="fas fa-history me-1"></i>View History
                </button>
                <button class="btn btn-outline-success btn-sm" onclick="analyzeAnother()">
                    <i class="fas fa-plus me-1"></i>Analyze Another
                </button>
            </div>
        </div>
    `;
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

function displayError(error) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="alert alert-danger mt-4">
            <i class="fas fa-exclamation-circle me-2"></i>
            <strong>Error:</strong> ${error}
        </div>
    `;
    resultDiv.style.display = 'block';
}

function analyzeAnother() {
    document.getElementById('text').value = '';
    document.getElementById('charCount').textContent = '0';
    document.getElementById('wordCount').textContent = '0';
    document.getElementById('result').style.display = 'none';
    document.getElementById('text').focus();
}
</script>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'analyze.html'), 'w', encoding='utf-8') as f:
        f.write(analyze_template)

    # History template
    history_template = '''{% extends "base.html" %}

{% block title %}Analysis History - FactForge{% endblock %}

{% block content %}
<div class="text-center mb-4">
    <h1 class="display-5 fw-bold text-primary">
        <i class="fas fa-history me-3"></i>Analysis History
    </h1>
    <p class="lead text-muted">Track your previous news analysis results</p>
</div>

<div class="row mb-4">
    <div class="col-md-8">
        <div class="card stats-card">
            <div class="card-body">
                <div class="row text-center">
                    <div class="col-md-3">
                        <h4 class="text-primary">{{ history|length }}</h4>
                        <small class="text-muted">Total Analyses</small>
                    </div>
                    <div class="col-md-3">
                        <h4 class="text-danger">{{ history|selectattr("prediction", "equalto", "FAKE")|list|length }}</h4>
                        <small class="text-muted">Fake News</small>
                    </div>
                    <div class="col-md-3">
                        <h4 class="text-success">{{ history|selectattr("prediction", "equalto", "REAL")|list|length }}</h4>
                        <small class="text-muted">Real News</small>
                    </div>
                    <div class="col-md-3">
                        {% set avg_conf = (history|selectattr("confidence", "ne", none)|map(attribute="confidence")|sum / (history|selectattr("confidence", "ne", none)|list|length)) if history|selectattr("confidence", "ne", none)|list|length > 0 else 0 %}
                        <h4 class="text-info">{{ "%.1f"|format(avg_conf * 100) }}%</h4>
                        <small class="text-muted">Avg Confidence</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="d-grid gap-2">
            <button class="btn btn-outline-danger" onclick="clearHistory()">
                <i class="fas fa-trash me-2"></i>Clear History
            </button>
            <a href="/api/export-history" class="btn btn-outline-primary" download="history.json">
                <i class="fas fa-download me-2"></i>Export History
            </a>
        </div>
    </div>
</div>

{% if history %}
<div class="card">
    <div class="card-header">
        <h5 class="mb-0">
            <i class="fas fa-list me-2"></i>Recent Analyses
        </h5>
    </div>
    <div class="card-body">
        {% for item in history %}
        <div class="history-item">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <div class="d-flex align-items-center mb-2">
                        <span class="badge bg-{{ 'danger' if item.prediction == 'FAKE' else 'success' }} me-2">
                            {{ item.prediction }}
                        </span>
                        {% if item.confidence %}
                        <small class="text-muted">{{ "%.1f"|format(item.confidence * 100) }}% confidence</small>
                        {% endif %}
                    </div>
                    <p class="mb-1">{{ item.text }}</p>
                    <small class="text-muted">
                        <i class="fas fa-clock me-1"></i>{{ item.timestamp }}
                    </small>
                </div>
                <div class="col-md-6">
                    <div class="text-end">
                        <small class="text-muted">
                            Words: {{ item.word_count }} → {{ item.processed_word_count }}<br>
                            <button class="btn btn-outline-primary btn-sm mt-1" onclick="viewFullText('{{ item.id }}')">
                                <i class="fas fa-eye me-1"></i>View Full
                            </button>
                        </small>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% else %}
<div class="text-center py-5">
    <i class="fas fa-inbox text-muted" style="font-size: 4rem;"></i>
    <h4 class="text-muted mt-3">No Analysis History</h4>
    <p class="text-muted">Start analyzing news articles to see your history here</p>
    <a href="/analyze" class="btn btn-primary">
        <i class="fas fa-search me-2"></i>Analyze Your First Article
    </a>
</div>
{% endif %}

<!-- Modal for full text view -->
<div class="modal fade" id="fullTextModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Full Article Text</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div id="fullTextContent"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
async function clearHistory() {
    if (confirm('Are you sure you want to clear all analysis history? This action cannot be undone.')) {
        try {
            const response = await fetch('/api/clear-history', { method: 'POST' });
            const data = await response.json();
            if (data.success) {
                location.reload();
            } else {
                alert('Error clearing history');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }
}

function viewFullText(itemId) {
    // This would need to be implemented with a proper backend endpoint
    // For now, show a placeholder
    document.getElementById('fullTextContent').innerHTML = '<p class="text-muted">Full text view would be implemented here</p>';
    new bootstrap.Modal(document.getElementById('fullTextModal')).show();
}
</script>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'history.html'), 'w', encoding='utf-8') as f:
        f.write(history_template)

    # Statistics template
    statistics_template = '''{% extends "base.html" %}

{% block title %}Statistics - FactForge{% endblock %}

{% block content %}
<div class="text-center mb-4">
    <h1 class="display-5 fw-bold text-primary">
        <i class="fas fa-chart-bar me-3"></i>Analysis Statistics
    </h1>
    <p class="lead text-muted">Detailed insights into your news analysis patterns</p>
</div>

<div class="row g-4 mb-4">
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-search text-primary" style="font-size: 2rem;"></i>
                <h3 class="mt-2 text-primary">{{ stats.total_predictions }}</h3>
                <p class="text-muted mb-0">Total Analyses</p>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-exclamation-triangle text-danger" style="font-size: 2rem;"></i>
                <h3 class="mt-2 text-danger">{{ stats.fake_count }}</h3>
                <p class="text-muted mb-0">Fake News Detected</p>
                <small class="text-muted">{{ "%.1f"|format(stats.fake_percentage) }}%</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-check-circle text-success" style="font-size: 2rem;"></i>
                <h3 class="mt-2 text-success">{{ stats.real_count }}</h3>
                <p class="text-muted mb-0">Real News Verified</p>
                <small class="text-muted">{{ "%.1f"|format(stats.real_percentage) }}%</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-chart-line text-info" style="font-size: 2rem;"></i>
                <h3 class="mt-2 text-info">{{ "%.1f"|format(stats.avg_confidence * 100) }}%</h3>
                <p class="text-muted mb-0">Avg Confidence</p>
            </div>
        </div>
    </div>
</div>

<div class="row g-4 mb-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-pie-chart me-2"></i>Prediction Distribution
                </h5>
            </div>
            <div class="card-body">
                <canvas id="distributionChart" width="400" height="200"></canvas>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-chart-area me-2"></i>Confidence Levels
                </h5>
            </div>
            <div class="card-body">
                <canvas id="confidenceChart" width="400" height="200"></canvas>
            </div>
        </div>
    </div>
</div>

{% if stats.recent_predictions %}
<div class="card">
    <div class="card-header">
        <h5 class="mb-0">
            <i class="fas fa-clock me-2"></i>Recent Analysis Trends
        </h5>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Text Preview</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Words</th>
                    </tr>
                </thead>
                <tbody>
                    {% for prediction in stats.recent_predictions %}
                    <tr>
                        <td>{{ prediction.timestamp }}</td>
                        <td>{{ prediction.text[:50] }}...</td>
                        <td>
                            <span class="badge bg-{{ 'danger' if prediction.prediction == 'FAKE' else 'success' }}">
                                {{ prediction.prediction }}
                            </span>
                        </td>
                        <td>
                            {% if prediction.confidence %}
                                {{ "%.1f"|format(prediction.confidence * 100) }}%
                            {% else %}
                                N/A
                            {% endif %}
                        </td>
                        <td>{{ prediction.word_count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endif %}

{% if stats.total_predictions == 0 %}
<div class="text-center py-5">
    <i class="fas fa-chart-bar text-muted" style="font-size: 4rem;"></i>
    <h4 class="text-muted mt-3">No Statistics Available</h4>
    <p class="text-muted">Start analyzing news articles to see detailed statistics</p>
    <a href="/analyze" class="btn btn-primary">
        <i class="fas fa-search me-2"></i>Start Analyzing
    </a>
</div>
{% endif %}
{% endblock %}

{% block scripts %}
<script>
// Distribution Chart
const distributionCtx = document.getElementById('distributionChart').getContext('2d');
new Chart(distributionCtx, {
    type: 'doughnut',
    data: {
        labels: ['Fake News', 'Real News'],
        datasets: [{
            data: [{{ stats.fake_count }}, {{ stats.real_count }}],
            backgroundColor: ['#dc2626', '#059669'],
            borderWidth: 2,
            borderColor: '#ffffff'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    }
});

// Confidence Chart (placeholder data for demonstration)
const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
new Chart(confidenceCtx, {
    type: 'bar',
    data: {
        labels: ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'],
        datasets: [{
            label: 'Predictions',
            data: [0, 1, 2, 3, 4], // This would be calculated from actual data
            backgroundColor: '#2563eb',
            borderRadius: 5
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 1
                }
            }
        },
        plugins: {
            legend: {
                display: false
            }
        }
    }
});
</script>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'statistics.html'), 'w', encoding='utf-8') as f:
        f.write(statistics_template)

    # Model details template
    model_details_template = '''{% extends "base.html" %}

{% block title %}Model Details - FactForge{% endblock %}

{% block content %}
<div class="text-center mb-4">
    <h1 class="display-5 fw-bold text-primary">
        <i class="fas fa-cogs me-3"></i>Model Details
    </h1>
    <p class="lead text-muted">Comprehensive information about the AI model</p>
</div>

{% if model_info %}
<div class="row g-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="fas fa-robot me-2"></i>Model Information
                </h5>
            </div>
            <div class="card-body">
                <table class="table table-borderless">
                    <tr>
                        <td><strong>Model Name:</strong></td>
                        <td>{{ model_info.model_name }}</td>
                    </tr>
                    <tr>
                        <td><strong>Algorithm:</strong></td>
                        <td>{{ model_info.model_type }}</td>
                    </tr>
                    <tr>
                        <td><strong>Training Date:</strong></td>
                        <td>{{ model_info.save_date[:10] }}</td>
                    </tr>
                    <tr>
                        <td><strong>Version:</strong></td>
                        <td>{{ model_info.save_timestamp }}</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>

    <div class="col-md-6">
        <div class="card">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0">
                    <i class="fas fa-chart-line me-2"></i>Performance Metrics
                </h5>
            </div>
            <div class="card-body">
                <div class="row text-center">
                    <div class="col-6 mb-3">
                        <h3 class="text-primary">{{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) }}%</h3>
                        <small class="text-muted">Accuracy</small>
                    </div>
                    <div class="col-6 mb-3">
                        <h3 class="text-success">{{ "%.3f"|format(model_info.performance_metrics.precision) }}</h3>
                        <small class="text-muted">Precision</small>
                    </div>
                    <div class="col-6">
                        <h3 class="text-info">{{ "%.3f"|format(model_info.performance_metrics.recall) }}</h3>
                        <small class="text-muted">Recall</small>
                    </div>
                    <div class="col-6">
                        <h3 class="text-warning">{{ "%.3f"|format(model_info.performance_metrics.f1_score) }}</h3>
                        <small class="text-muted">F1-Score</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row g-4 mt-2">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-info-circle me-2"></i>How the Model Works
                </h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6><i class="fas fa-cog me-2"></i>Text Preprocessing</h6>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check text-success me-2"></i>Lowercase conversion</li>
                            <li><i class="fas fa-check text-success me-2"></i>Stopword removal (217 words)</li>
                            <li><i class="fas fa-check text-success me-2"></i>Punctuation & digit removal</li>
                            <li><i class="fas fa-check text-success me-2"></i>Tokenization & lemmatization</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6><i class="fas fa-search me-2"></i>Feature Extraction</h6>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check text-success me-2"></i>TF-IDF Vectorization</li>
                            <li><i class="fas fa-check text-success me-2"></i>N-gram analysis (1-2 grams)</li>
                            <li><i class="fas fa-check text-success me-2"></i>Feature selection</li>
                            <li><i class="fas fa-check text-success me-2"></i>Dimensionality optimization</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row g-4 mt-2">
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-database text-primary" style="font-size: 2rem;"></i>
                <h6 class="mt-2">Training Data</h6>
                <p class="text-muted small">Balanced dataset with equal FAKE and REAL news samples</p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-brain text-success" style="font-size: 2rem;"></i>
                <h6 class="mt-2">Algorithm</h6>
                <p class="text-muted small">{{ model_info.model_type }} with optimized hyperparameters</p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-shield-alt text-info" style="font-size: 2rem;"></i>
                <h6 class="mt-2">Validation</h6>
                <p class="text-muted small">5-fold cross-validation for robust performance</p>
            </div>
        </div>
    </div>
</div>
{% else %}
<div class="text-center py-5">
    <i class="fas fa-exclamation-triangle text-warning" style="font-size: 4rem;"></i>
    <h4 class="text-muted mt-3">No Model Information Available</h4>
    <p class="text-muted">Model details could not be loaded</p>
</div>
{% endif %}
{% endblock %}'''

    with open(os.path.join(templates_dir, 'model_details.html'), 'w', encoding='utf-8') as f:
        f.write(model_details_template)
    
    # Index template
    index_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <h1 class="text-center mb-4">🔍 Fake News Detection</h1>
        <p class="text-center text-muted">Enter news text below to check if it's likely to be fake or real</p>
        
        {% if model_info %}
        <div class="alert alert-info">
            <strong>Model:</strong> {{ model_info.model_name }} | 
            <strong>Accuracy:</strong> {{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) }}% |
            <strong>F1-Score:</strong> {{ "%.3f"|format(model_info.performance_metrics.f1_score) }}
        </div>
        {% endif %}
        
        <form id="predictionForm">
            <div class="mb-3">
                <label for="text" class="form-label">News Text:</label>
                <textarea class="form-control" id="text" name="text" rows="6" 
                         placeholder="Enter the news article text here..." required></textarea>
                <div class="form-text">Minimum 10 characters required</div>
            </div>
            <button type="submit" class="btn btn-primary btn-lg w-100">
                <span id="submitText">Analyze Text</span>
                <span id="loadingSpinner" class="spinner-border spinner-border-sm ms-2" style="display: none;"></span>
            </button>
        </form>
        
        <div id="result" style="display: none;"></div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultDiv = document.getElementById('result');
    
    // Show loading state
    submitBtn.disabled = true;
    submitText.textContent = 'Analyzing...';
    loadingSpinner.style.display = 'inline-block';
    resultDiv.style.display = 'none';
    
    try {
        const formData = new FormData(e.target);
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
        } else {
            displayError(data.error);
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitText.textContent = 'Analyze Text';
        loadingSpinner.style.display = 'none';
    }
});

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const isFake = data.prediction === 'FAKE';
    const confidence = data.confidence || 0.5;
    
    resultDiv.innerHTML = `
        <div class="prediction-result ${isFake ? 'fake-result' : 'real-result'}">
            <h3>${isFake ? '🚨' : '✅'} ${data.prediction}</h3>
            ${data.confidence ? `
                <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill bg-${isFake ? 'danger' : 'success'}" 
                         style="width: ${confidence * 100}%"></div>
                </div>
            ` : ''}
            <hr>
            <small class="text-muted">
                Original: ${data.word_count} words | 
                Processed: ${data.processed_word_count} words | 
                Reduction: ${(data.reduction_ratio * 100).toFixed(1)}%
            </small>
        </div>
    `;
    resultDiv.style.display = 'block';
}

function displayError(error) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="alert alert-danger">
            <strong>Error:</strong> ${error}
        </div>
    `;
    resultDiv.style.display = 'block';
}
</script>
{% endblock %}'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_template)

    # Enhanced batch template
    batch_template = '''{% extends "base.html" %}

{% block title %}Batch Analysis - FactForge{% endblock %}

{% block content %}
<div class="text-center mb-4">
    <h1 class="display-5 fw-bold text-primary">
        <i class="fas fa-layer-group me-3"></i>Batch Analysis
    </h1>
    <p class="lead text-muted">Analyze multiple news articles simultaneously for efficient processing</p>
</div>

<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="fas fa-upload me-2"></i>Batch News Analysis
                </h5>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-4">
                        <label for="texts" class="form-label fw-bold">
                            <i class="fas fa-list me-2"></i>News Articles (one per line)
                        </label>
                        <textarea class="form-control" id="texts" name="texts" rows="12"
                                 placeholder="Enter news articles here, one per line...

Example:
Scientists discover new planet made of chocolate
Federal Reserve announces interest rate decision
Local man claims pet goldfish speaks 17 languages"
                                 required style="border-radius: 10px; border: 2px solid #e5e7eb;"></textarea>
                        <div class="form-text">
                            <i class="fas fa-info-circle me-1"></i>
                            Enter up to 50 news articles, one per line. Each article should be at least 10 characters.
                        </div>
                        <div class="mt-2">
                            <small class="text-muted">
                                Lines: <span id="lineCount">0</span> |
                                Total Characters: <span id="totalChars">0</span>
                            </small>
                        </div>
                    </div>

                    <div class="d-grid">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-cogs me-2"></i>Analyze All Articles
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <i class="fas fa-rocket text-primary" style="font-size: 2rem;"></i>
                        <h6 class="mt-2">Fast Processing</h6>
                        <small class="text-muted">Analyze up to 50 articles in seconds</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <i class="fas fa-download text-success" style="font-size: 2rem;"></i>
                        <h6 class="mt-2">Export Results</h6>
                        <small class="text-muted">Download results as CSV or JSON</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('texts').addEventListener('input', function() {
    const text = this.value;
    const lines = text.split('\n').filter(line => line.trim().length > 0);
    document.getElementById('lineCount').textContent = lines.length;
    document.getElementById('totalChars').textContent = text.length;

    // Warn if too many lines
    if (lines.length > 50) {
        this.style.borderColor = '#dc2626';
        document.querySelector('.form-text').innerHTML =
            '<i class="fas fa-exclamation-triangle text-danger me-1"></i>Too many articles! Maximum 50 allowed.';
    } else {
        this.style.borderColor = '#e5e7eb';
        document.querySelector('.form-text').innerHTML =
            '<i class="fas fa-info-circle me-1"></i>Enter up to 50 news articles, one per line. Each article should be at least 10 characters.';
    }
});
</script>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'batch.html'), 'w', encoding='utf-8') as f:
        f.write(batch_template)

    # Batch results template
    batch_results_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-10 mx-auto">
        <h1 class="text-center mb-4">📊 Batch Analysis Results</h1>

        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">{{ summary.total }}</h5>
                        <p class="card-text">Total Texts</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center bg-danger text-white">
                    <div class="card-body">
                        <h5 class="card-title">{{ summary.fake }}</h5>
                        <p class="card-text">FAKE ({{ "%.1f"|format(summary.fake_percentage) }}%)</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center bg-success text-white">
                    <div class="card-body">
                        <h5 class="card-title">{{ summary.real }}</h5>
                        <p class="card-text">REAL ({{ "%.1f"|format(summary.real_percentage) }}%)</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <a href="/batch" class="btn btn-primary btn-lg h-100 d-flex align-items-center justify-content-center">
                    Analyze More
                </a>
            </div>
        </div>

        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Text Preview</th>
                        <th>Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr>
                        <td>{{ result.index }}</td>
                        <td>{{ result.text_preview }}</td>
                        <td>
                            <span class="badge bg-{{ 'danger' if result.prediction == 'FAKE' else 'success' }}">
                                {{ result.prediction }}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'batch_results.html'), 'w', encoding='utf-8') as f:
        f.write(batch_results_template)

    # About template
    about_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <h1 class="text-center mb-4">ℹ️ About</h1>

        <div class="card mb-4">
            <div class="card-header">
                <h5>🤖 Model Information</h5>
            </div>
            <div class="card-body">
                {% if model_info %}
                <p><strong>Model Name:</strong> {{ model_info.model_name }}</p>
                <p><strong>Model Type:</strong> {{ model_info.model_type }}</p>
                <p><strong>Training Date:</strong> {{ model_info.save_date }}</p>
                <h6>Performance Metrics:</h6>
                <ul>
                    <li>Accuracy: {{ "%.1f"|format(model_info.performance_metrics.accuracy * 100) }}%</li>
                    <li>Precision: {{ "%.3f"|format(model_info.performance_metrics.precision) }}</li>
                    <li>Recall: {{ "%.3f"|format(model_info.performance_metrics.recall) }}</li>
                    <li>F1-Score: {{ "%.3f"|format(model_info.performance_metrics.f1_score) }}</li>
                </ul>
                {% else %}
                <p>No model information available.</p>
                {% endif %}
            </div>
        </div>

        <div class="card mb-4">
            <div class="card-header">
                <h5>📚 How It Works</h5>
            </div>
            <div class="card-body">
                <ol>
                    <li><strong>Text Preprocessing:</strong> The input text is cleaned, tokenized, and processed using NLTK</li>
                    <li><strong>Feature Extraction:</strong> TF-IDF vectorization converts text into numerical features</li>
                    <li><strong>Classification:</strong> Machine learning model predicts FAKE or REAL</li>
                    <li><strong>Confidence Score:</strong> Probability score indicates prediction certainty</li>
                </ol>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h5>⚠️ Disclaimer</h5>
            </div>
            <div class="card-body">
                <p>This tool is for educational and research purposes. The predictions are based on machine learning
                models trained on sample data and should not be considered as definitive truth. Always verify
                information from multiple reliable sources.</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''

    with open(os.path.join(templates_dir, 'about.html'), 'w', encoding='utf-8') as f:
        f.write(about_template)


if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting...")
        sys.exit(1)
    
    print("🚀 Starting Flask web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
