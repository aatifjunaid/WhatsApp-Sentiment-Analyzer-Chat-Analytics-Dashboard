"""Complete Training Pipeline"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import ClassicalMLModels, ModelEvaluator
from config import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, OUTPUTS_DIR, RANDOM_STATE


class TrainingPipeline:
    """Complete training pipeline from data to model"""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("WHATSAPP SENTIMENT ANALYSIS - FULL TRAINING PIPELINE")
        print("="*70)
        
        # Phase 1: Data Loading
        print("\n[PHASE 1] Data Acquisition & Exploration...")
        self.load_and_explore_data()
        
        # Phase 2: Preprocessing
        print("\n[PHASE 2] Text Preprocessing...")
        self.preprocess_data()
        
        # Phase 3: Feature Extraction
        print("\n[PHASE 3] Feature Extraction...")
        self.extract_features()
        
        # Phase 4: Train-Test Split
        print("\n[PHASE 4] Creating Train/Val/Test Split...")
        self.split_data()
        
        # Phase 5: Model Training
        print("\n[PHASE 5] Model Training & Selection...")
        self.train_models()
        
        # Phase 6: Evaluation
        print("\n[PHASE 6] Model Evaluation...")
        self.evaluate_models()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)
    
    def load_and_explore_data(self):
        """Load and explore dataset"""
        loader = DataLoader()
        
        if loader.load_data():
            self.data = loader.data
            stats = loader.explore_data()
            loader.visualize_distribution(
                save_path=OUTPUTS_DIR / "eda_distribution.png"
            )
            print(f"✓ Loaded {len(self.data)} records")
            return True
        return False
    
    def preprocess_data(self):
        """Preprocess text data"""
        if self.data is None:
            print("✗ Data not loaded")
            return False
        
        preprocessor = TextPreprocessor(normalize_emojis=True, remove_stopwords=False)
        
        print("  Preprocessing texts...")
        self.data['processed_text'] = preprocessor.preprocess_batch(
            self.data['text'].tolist(),
            full_pipeline=True
        )
        
        print("  Sample preprocessed text:")
        for i in range(min(3, len(self.data))):
            print(f"    Original: {self.data['text'].iloc[i][:60]}...")
            print(f"    Processed: {self.data['processed_text'].iloc[i][:60]}...")
            print()
        
        print(f"✓ Preprocessed {len(self.data)} texts")
        return True
    
    def extract_features(self):
        """Extract features from preprocessed text"""
        if self.data is None:
            print("✗ Data not preprocessed")
            return False
        
        extractor = FeatureExtractor()
        
        # TF-IDF features
        print("  Creating TF-IDF features...")
        self.tfidf_features = extractor.tfidf_vectorization(
            self.data['processed_text'].tolist(),
            max_features=5000,
            ngram_range=(1, 2),
            fit=True
        )
        
        self.tfidf_vectorizer = extractor.tfidf_vectorizer
        
        print(f"✓ TF-IDF features shape: {self.tfidf_features.shape}")
        return True
    
    def split_data(self):
        """Split data into train, validation, test"""
        from sklearn.model_selection import train_test_split
        
        n_samples = len(self.data)
        train_size = int(n_samples * TRAIN_SIZE)
        temp_size = n_samples - train_size
        val_size = int(temp_size * (VAL_SIZE / (VAL_SIZE + TEST_SIZE)))
        
        # Split features and labels
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.tfidf_features,
            self.data['airline_sentiment'],
            test_size=1-TRAIN_SIZE,
            random_state=RANDOM_STATE,
            stratify=self.data['airline_sentiment']
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        print(f"  Train: {len(self.y_train)} ({TRAIN_SIZE*100:.0f}%)")
        print(f"  Val:   {len(self.y_val)} ({VAL_SIZE*100:.0f}%)")
        print(f"  Test:  {len(self.y_test)} ({TEST_SIZE*100:.0f}%)")
        
        # Print class distribution
        print("\n  Class distribution (Train):")
        for sentiment, count in self.y_train.value_counts().items():
            pct = count / len(self.y_train) * 100
            print(f"    {sentiment}: {count} ({pct:.1f}%)")
    
    def train_models(self):
        """Train multiple models"""
        ml_models = ClassicalMLModels()
        
        # 1. Logistic Regression
        print("  Training Logistic Regression...")
        lr_model = ml_models.logistic_regression_model(
            self.X_train, self.y_train, C=1.0
        )
        self.models['logistic_regression'] = lr_model
        
        # 2. SVM
        print("  Training SVM...")
        svm_model = ml_models.svm_model(
            self.X_train, self.y_train, kernel='rbf', C=1.0
        )
        self.models['svm'] = svm_model
        
        print(f"✓ Trained {len(self.models)} models")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        evaluator = ModelEvaluator()
        
        for model_name, model in self.models.items():
            print(f"\n  Evaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            results = evaluator.evaluate_predictions(self.y_test, y_pred)
            self.results[model_name] = results
            
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    F1-Score (weighted): {results['f1_weighted']:.4f}")
            
            # Save results
            results_path = OUTPUTS_DIR / f"{model_name}_results.json"
            evaluator.save_results(results_path)
            
            # Save confusion matrix
            cm_path = OUTPUTS_DIR / f"{model_name}_confusion_matrix.png"
            evaluator.plot_confusion_matrix(self.y_test, y_pred, save_path=cm_path)
        
        # Select best model
        best_model_name = max(self.results, 
                             key=lambda x: self.results[x]['f1_weighted'])
        print(f"\n✓ Best model: {best_model_name} "
              f"(F1: {self.results[best_model_name]['f1_weighted']:.4f})")
        
        return best_model_name
    
    def save_best_model(self, model_name):
        """Save best model"""
        import joblib
        from config import MODELS_DIR
        
        model = self.models[model_name]
        model_path = MODELS_DIR / f"{model_name}_final.pkl"
        
        joblib.dump(model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Also save vectorizer
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        joblib.dump(self.tfidf_vectorizer, vectorizer_path)
        print(f"✓ Vectorizer saved to {vectorizer_path}")


def main():
    """Main execution"""
    pipeline = TrainingPipeline()
    pipeline.run_full_pipeline()
    
    # Save best model
    best_model = max(pipeline.results, 
                    key=lambda x: pipeline.results[x]['f1_weighted'])
    pipeline.save_best_model(best_model)
    
    print("\n" + "="*70)
    print("Training complete! Check outputs/ folder for results.")
    print("="*70)


if __name__ == "__main__":
    main()
