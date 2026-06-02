"""Configuration settings for WhatsApp Sentiment Analysis Project"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data configuration
DATASET_NAME = "crowdflower/twitter-airline-sentiment"
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Model configuration
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
NUM_CLASSES = 3

# Training configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
DEVICE = "cuda"  # or "cpu"

# Model checkpoint names
LOGISTIC_REGRESSION_MODEL = MODELS_DIR / "logistic_regression.pkl"
SVM_MODEL = MODELS_DIR / "svm_model.pkl"
LSTM_MODEL = MODELS_DIR / "lstm_model.pt"
DISTILBERT_MODEL = MODELS_DIR / "distilbert_model"

# Preprocessing config
STOPWORDS_LANG = "english"
LOWERCASE = True
REMOVE_URLS = True
REMOVE_MENTIONS = True
NORMALIZE_EMOJIS = True

# Slang and abbreviation expansions
SLANG_DICT = {
    "u": "you",
    "ur": "your",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "tbh": "to be honest",
    "idk": "i don't know",
    "btw": "by the way",
    "fyi": "for your information",
    "imo": "in my opinion",
    "smh": "shaking my head",
}
