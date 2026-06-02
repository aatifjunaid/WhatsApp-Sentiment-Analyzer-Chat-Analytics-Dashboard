# Google Colab Notebook - WhatsApp Sentiment Analysis Training
# Ready-to-copy code cells for Google Colab

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
!pip install -q pandas numpy matplotlib seaborn nltk spacy scikit-learn emoji tqdm joblib
!python -m spacy download en_core_web_sm
print("✓ Dependencies installed successfully!")


# ============================================================================
# CELL 2: Clone Project from GitHub (or upload ZIP)
# ============================================================================
# CHOOSE ONE OPTION:

# OPTION A: Clone from GitHub (Recommended)
# ========================================
# First create a GitHub repo and push all your project files
# Then replace YOUR_USERNAME with your actual GitHub username

# !git clone https://github.com/YOUR_USERNAME/whatsapp-sentiment-analysis.git
# %cd whatsapp-sentiment-analysis

# OPTION B: Upload as ZIP file
# =============================
from google.colab import files
print("Upload your project as a ZIP file (all code and data)")
uploaded = files.upload()

# Get the ZIP filename
import zipfile
zip_file = [f for f in uploaded.keys() if f.endswith('.zip')][0]
print(f"\nExtracting {zip_file}...")

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()

# Navigate to project folder (adjust folder name if different)
import os
folders = [f for f in os.listdir() if os.path.isdir(f) and f != '__pycache__']
if folders:
    project_folder = folders[0]
    print(f"Found project folder: {project_folder}")
    os.chdir(project_folder)

print("\nProject files:")
os.system('ls -la')


# ============================================================================
# CELL 3: Check GPU
# ============================================================================
import torch

print("GPU Information:")
print(f"  GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("\n⚠️  No GPU detected!")
    print("Fix: Go to Runtime → Change runtime type → GPU")


# ============================================================================
# CELL 4: Import Modules
# ============================================================================
import sys
sys.path.insert(0, '/content')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try importing project modules
try:
    from src.data_loader import DataLoader
    from src.preprocessor import TextPreprocessor
    from src.feature_extraction import FeatureExtractor
    from src.models import ClassicalMLModels
    from src.evaluation import ModelEvaluator
    from src.inference import SentimentPredictor
    from train_pipeline import TrainingPipeline
    print("✓ All project modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure project structure is correct:")
    print("  src/")
    print("  ├── data_loader.py")
    print("  ├── preprocessor.py")
    print("  ├── feature_extraction.py")
    print("  ├── models.py")
    print("  ├── evaluation.py")
    print("  └── inference.py")


# ============================================================================
# CELL 5: Mount Google Drive (to save results)
# ============================================================================
from google.colab import drive

drive.mount('/content/drive')

# Create saving directories
!mkdir -p '/content/drive/MyDrive/sentiment-analysis/models'
!mkdir -p '/content/drive/MyDrive/sentiment-analysis/results'
!mkdir -p '/content/drive/MyDrive/sentiment-analysis/data'

print("✓ Google Drive mounted!")
print("✓ Directories created")


# ============================================================================
# CELL 6: Get Dataset from Kaggle
# ============================================================================
# CHOOSE ONE OPTION:

# OPTION A: Use Kaggle API (Automatic) - RECOMMENDED
# ==================================================
!pip install -q kaggle

from google.colab import files
print("1. Download kaggle.json from: https://www.kaggle.com/settings/account")
print("2. Upload the file here:")

uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("\n📥 Downloading dataset from Kaggle...")
!kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data/
!unzip -q data/twitter-airline-sentiment.zip -d data/

print("\n✓ Dataset ready!")
os.system('ls -lh data/Tweets.csv')

# OPTION B: Manual Download
# ==========================
# 1. Visit: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
# 2. Download Tweets.csv
# 3. Upload CSV to Colab

# from google.colab import files
# print("Upload Tweets.csv file")
# uploaded = files.upload()
# !mv Tweets.csv data/


# ============================================================================
# CELL 7: Quick Data Exploration
# ============================================================================
print("📊 Loading and exploring dataset...\n")

# Load data
from config import DATA_PATH
df = pd.read_csv('data/Tweets.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nSentiment distribution:")
print(df['airline_sentiment'].value_counts())

print(f"\nBasic statistics:")
print(f"  - Total tweets: {len(df)}")
print(f"  - Average length: {df['text'].str.len().mean():.0f} characters")
print(f"  - Max length: {df['text'].str.len().max()} characters")


# ============================================================================
# CELL 8: Run Complete Training Pipeline
# ============================================================================
print("🚀 STARTING TRAINING PIPELINE\n")
print("="*70)

from train_pipeline import TrainingPipeline

# Initialize and run pipeline
pipeline = TrainingPipeline()

try:
    pipeline.run_full_pipeline()
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# CELL 9: Review Training Results
# ============================================================================
print("📊 TRAINING RESULTS\n")

# Check results
if hasattr(pipeline, 'results') and pipeline.results:
    results_df = pd.DataFrame(pipeline.results).T
    print("Model Performance Summary:\n")
    print(results_df[['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']].to_string())
    
    # Find best model
    best_model = results_df['f1_weighted'].idxmax()
    best_score = results_df['f1_weighted'].max()
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1-Score: {best_score:.4f}")
else:
    print("Check outputs/ folder for results")


# ============================================================================
# CELL 10: Save Models to Google Drive
# ============================================================================
import shutil

print("💾 Saving to Google Drive...\n")

# Save models
if os.path.exists('models/'):
    drive_dest = '/content/drive/MyDrive/sentiment-analysis/models'
    shutil.copytree('models/', drive_dest, dirs_exist_ok=True)
    print(f"✓ Models saved\n  Location: {drive_dest}")
    print(f"\nSaved models:")
    os.system('ls -lh models/')

# Save results
if os.path.exists('outputs/'):
    drive_dest = '/content/drive/MyDrive/sentiment-analysis/results'
    shutil.copytree('outputs/', drive_dest, dirs_exist_ok=True)
    print(f"\n✓ Results saved\n  Location: {drive_dest}")

print("\n✅ All files saved!")


# ============================================================================
# CELL 11: Test Predictions on Example Texts
# ============================================================================
print("🧪 Testing Predictions\n")

# Load best model
from src.inference import SentimentPredictor

model_path = 'models/logistic_regression_final.pkl'
predictor = SentimentPredictor(model_path)

# Test samples
test_samples = [
    "I absolutely LOVE this airline! Best service ever! 😍",
    "This is amazing! Very happy with my experience",
    "It was okay, nothing special",
    "Not great, but acceptable",
    "Worst experience ever! Terrible service! 😡",
    "I hate flying with them. Never again!",
]

print("Making predictions...\n")

results = predictor.predict_batch(test_samples, return_proba=True)

# Display results
for i, (text, result) in enumerate(zip(test_samples, results), 1):
    sent = result['sentiment'].upper()
    conf = max(result['probabilities'].values())
    text_preview = text[:40] + "..." if len(text) > 40 else text
    print(f"{i}. [{sent}] {text_preview}")
    print(f"   Confidence: {conf:.1%}\n")


# ============================================================================
# CELL 12 (OPTIONAL): Analyze WhatsApp Chat
# ============================================================================
# Uncomment and run if you want to analyze a WhatsApp chat export

"""
print("📱 WhatsApp Chat Analysis\n")

from google.colab import files
print("Upload your WhatsApp chat export (.txt file)")
uploaded = files.upload()

if uploaded:
    chat_file = list(uploaded.keys())[0]
    
    # Analyze
    predictor = SentimentPredictor('models/logistic_regression_final.pkl')
    df_chat = predictor.predict_from_file(chat_file, output_path='whatsapp_results.csv')
    
    print("\nAnalysis results:")
    print(df_chat.head(10))
    
    print("\nSentiment distribution:")
    print(df_chat['sentiment'].value_counts())
    
    # Download
    files.download('whatsapp_results.csv')
"""


# ============================================================================
# CELL 13: Download All Results
# ============================================================================
from google.colab import files

print("Downloading files...\n")

# Download models
print("Models:")
for file in os.listdir('models/'):
    print(f"  ✓ {file}")
    files.download(f'models/{file}')

# Download results
print("\nResults:")
for file in os.listdir('outputs/'):
    if file.endswith(('.json', '.png', '.csv')):
        print(f"  ✓ {file}")
        files.download(f'outputs/{file}')

print("\n✅ All downloads complete!")
