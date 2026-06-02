"""
Setup and initialization script for WhatsApp Sentiment Analysis Project
Run this first to set up the environment
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required. You have {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Create required directories"""
    dirs = ['data', 'models', 'outputs', 'notebooks']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Directory '{dir_name}' created/verified")

def install_dependencies():
    """Install required packages"""
    print("\n[INSTALLING DEPENDENCIES]")
    print("This may take a few minutes...\n")
    
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\n✓ All dependencies installed")

def download_spacy_model():
    """Download spaCy English model"""
    print("\n[DOWNLOADING NLP MODELS]")
    
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✓ spaCy model already installed")
    except OSError:
        print("Downloading spaCy English model...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded")

def download_nltk_data():
    """Download NLTK data"""
    print("\n[DOWNLOADING NLTK DATA]")
    
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK punkt tokenizer already present")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ NLTK stopwords already present")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

def configure_kaggle():
    """Setup Kaggle API (optional)"""
    print("\n[KAGGLE SETUP (Optional)]")
    print("""
To download dataset automatically:
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save kaggle.json to ~/.kaggle/ (or C:\\Users\\<YourUsername>\\.kaggle\\ on Windows)
4. Run: kaggle datasets download -d crowdflower/twitter-airline-sentiment

Or download manually from:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
and save Tweets.csv to data/ folder
""")

def verify_installation():
    """Verify all installations"""
    print("\n[VERIFYING INSTALLATION]")
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError:
        print("✗ pandas - install failed")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - install failed")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        print("✗ scikit-learn - install failed")
        return False
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError:
        print("✗ nltk - install failed")
        return False
    
    try:
        import spacy
        print("✓ spacy")
    except ImportError:
        print("✗ spacy - install failed")
        return False
    
    try:
        import emoji
        print("✓ emoji")
    except ImportError:
        print("✗ emoji - install failed")
        return False
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError:
        print("⚠ transformers - optional (for BERT models)")
    
    try:
        import torch
        print("✓ torch")
    except ImportError:
        print("⚠ torch - optional (for deep learning models)")
    
    return True

def main():
    """Run setup"""
    print("="*70)
    print("WHATSAPP SENTIMENT ANALYSIS - PROJECT SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n[CREATING DIRECTORIES]")
    create_directories()
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        print("\nTry installing manually:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Download NLP models
    try:
        download_nltk_data()
    except Exception as e:
        print(f"⚠ Warning: {e}")
    
    try:
        download_spacy_model()
    except Exception as e:
        print(f"⚠ Warning: {e}")
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Some packages failed to install")
        sys.exit(1)
    
    # Kaggle setup
    configure_kaggle()
    
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE!")
    print("="*70)
    
    print("""
NEXT STEPS:

1. Download the dataset:
   - Automatic: kaggle datasets download -d crowdflower/twitter-airline-sentiment
   - Manual: Download from Kaggle and save Tweets.csv to data/ folder

2. Run the training pipeline:
   python train_pipeline.py

3. Check examples:
   python examples.py

4. Read documentation:
   cat README.md

5. Or run individual components:
   from src.data_loader import DataLoader
   loader = DataLoader()
   loader.load_data()
   loader.explore_data()

Questions? See README.md or examples.py
""")

if __name__ == "__main__":
    main()
