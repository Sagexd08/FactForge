"""
Setup Script for Fake News Detection Project

This script helps users set up the project by installing dependencies
and downloading required NLTK data.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install Python packages from requirements.txt"""
    print("📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        
        # Download required NLTK data
        nltk_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        
        for package in nltk_packages:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False


def download_spacy_model():
    """Download SpaCy English model (optional)"""
    print("🔤 Downloading SpaCy English model (optional)...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ SpaCy model downloaded successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  SpaCy model download failed (optional - you can skip this)")
        return False


def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "models/saved_models",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directories created successfully!")


def main():
    """Main setup function"""
    print("🚀 Setting up Fake News Detection Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed at NLTK data download")
        sys.exit(1)
    
    # Download SpaCy model (optional)
    download_spacy_model()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main.py' to train the model")
    print("2. Run 'python cli_interface.py' for command line interface")
    print("3. Run 'python web_app.py' for web interface")
    print("=" * 50)


if __name__ == "__main__":
    main()
