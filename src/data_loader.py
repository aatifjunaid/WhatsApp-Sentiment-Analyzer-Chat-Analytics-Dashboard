"""Phase 1: Data Acquisition & Exploration"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, DATASET_NAME, SENTIMENT_LABELS

class DataLoader:
    """Handle data acquisition and exploratory data analysis"""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def download_dataset(self):
        """Download dataset from Kaggle using kaggle API"""
        print(f"Downloading {DATASET_NAME} from Kaggle...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)
            print(f"Dataset downloaded to {DATA_DIR}")
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            print("Please download manually from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment")
            return False
        return True
    
    def load_data(self, csv_path=None):
        """Load the airline sentiment dataset"""
        if csv_path is None:
            csv_path = DATA_DIR / "Tweets.csv"
        
        if not Path(csv_path).exists():
            print(f"Dataset not found at {csv_path}")
            if self.download_dataset():
                csv_path = DATA_DIR / "Tweets.csv"
            else:
                return False
        
        print(f"Loading data from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} records")
        return True
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nDataset Shape:", self.data.shape)
        print("\nColumn Names:", self.data.columns.tolist())
        print("\nData Types:\n", self.data.dtypes)
        print("\nMissing Values:\n", self.data.isnull().sum())
        
        # Sentiment distribution
        print("\nSentiment Distribution:")
        sentiment_counts = self.data['airline_sentiment'].value_counts()
        print(sentiment_counts)
        print("\nSentiment Distribution (%):")
        print((sentiment_counts / len(self.data) * 100).round(2))
        
        # Text statistics
        self.data['text_length'] = self.data['text'].str.len()
        self.data['word_count'] = self.data['text'].str.split().str.len()
        
        print("\nText Statistics:")
        print(f"Mean text length: {self.data['text_length'].mean():.2f} chars")
        print(f"Max text length: {self.data['text_length'].max()} chars")
        print(f"Min text length: {self.data['text_length'].min()} chars")
        print(f"Mean word count: {self.data['word_count'].mean():.2f} words")
        
        # Sample texts
        print("\nSample Texts:")
        for sentiment in self.data['airline_sentiment'].unique():
            print(f"\n{sentiment.upper()}:")
            sample = self.data[self.data['airline_sentiment'] == sentiment]['text'].iloc[0]
            print(f"  {sample[:100]}...")
        
        return sentiment_counts
    
    def visualize_distribution(self, save_path=None):
        """Visualize sentiment distribution"""
        if self.data is None:
            print("Data not loaded.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sentiment distribution
        sentiment_counts = self.data['airline_sentiment'].value_counts()
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'gray', 'green'])
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Text length by sentiment
        self.data.boxplot(column='text_length', by='airline_sentiment', ax=axes[1])
        axes[1].set_title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sentiment')
        axes[1].set_ylabel('Text Length (characters)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        plt.show()
    
    def get_statistics_summary(self):
        """Return a summary of dataset statistics"""
        if self.data is None:
            return None
        
        summary = {
            'total_records': len(self.data),
            'sentiment_distribution': self.data['airline_sentiment'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': self.data['text_length'].mean(),
                'min': self.data['text_length'].min(),
                'max': self.data['text_length'].max(),
            },
            'word_count_stats': {
                'mean': self.data['word_count'].mean(),
                'min': self.data['word_count'].min(),
                'max': self.data['word_count'].max(),
            }
        }
        return summary


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    if loader.load_data():
        loader.explore_data()
        loader.visualize_distribution(save_path=Path("../outputs/eda_visualization.png"))
