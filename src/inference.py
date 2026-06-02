"""Phase 6: Deployment and Inference"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENTIMENT_LABELS, MODELS_DIR
from preprocessor import TextPreprocessor, WhatsAppParser
from feature_extraction import FeatureExtractor


class SentimentPredictor:
    """Load and use trained models for inference"""
    
    def __init__(self, model_path, model_type='classical'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model
            model_type: 'classical' for sklearn, 'transformer' for HuggingFace
        """
        self.model = joblib.load(model_path)
        self.model_type = model_type
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        print(f"Model loaded from {model_path}")
    
    def predict_single(self, text, return_proba=False):
        """
        Predict sentiment for single text
        
        Args:
            text: Input text string
            return_proba: Whether to return probabilities
        
        Returns:
            sentiment label or tuple of (label, probabilities)
        """
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text, full_pipeline=True)
        
        # Vectorize
        features = self.feature_extractor.tfidf_vectorization(
            [cleaned_text], fit=False
        )
        
        # Predict
        prediction = self.model.predict(features)[0]
        sentiment = SENTIMENT_LABELS[prediction]
        
        if return_proba:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                return sentiment, dict(zip(SENTIMENT_LABELS, proba))
            else:
                return sentiment, None
        
        return sentiment
    
    def predict_batch(self, texts, return_proba=False):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of text strings
            return_proba: Whether to return probabilities
        
        Returns:
            List of predictions
        """
        results = []
        for text in texts:
            if return_proba:
                pred, proba = self.predict_single(text, return_proba=True)
                results.append({'text': text, 'sentiment': pred, 'probabilities': proba})
            else:
                pred = self.predict_single(text, return_proba=False)
                results.append({'text': text, 'sentiment': pred})
        
        return results
    
    def predict_from_file(self, file_path, output_path=None):
        """
        Predict sentiment from WhatsApp chat export
        
        Args:
            file_path: Path to .txt WhatsApp export
            output_path: Path to save CSV results
        
        Returns:
            DataFrame with predictions
        """
        parser = WhatsAppParser()
        messages = parser.parse_whatsapp_export(file_path)
        
        if not messages:
            print("No messages parsed from file")
            return None
        
        print(f"Parsed {len(messages)} messages")
        
        # Predict sentiment for each message
        results = []
        for msg in messages:
            sentiment, proba = self.predict_single(msg['message'], return_proba=True)
            results.append({
                'date': msg['date'],
                'time': msg['time'],
                'sender': msg['sender'],
                'message': msg['message'],
                'sentiment': sentiment,
                'confidence': max(proba.values()) if proba else None
            })
        
        df = pd.DataFrame(results)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return df


class BatchInferenceEngine:
    """Process multiple messages efficiently"""
    
    def __init__(self, model_path, batch_size=32):
        self.predictor = SentimentPredictor(model_path)
        self.batch_size = batch_size
    
    def process_dataframe(self, df, text_column='text'):
        """
        Process entire DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column name containing text
        
        Returns:
            DataFrame with sentiment predictions
        """
        texts = df[text_column].tolist()
        
        predictions = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_preds = self.predictor.predict_batch(batch, return_proba=True)
            predictions.extend(batch_preds)
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {**p, 'text': p['text']} for p in predictions
        ])
        
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)


def create_inference_pipeline(model_path, input_file=None):
    """
    Quick inference function
    
    Args:
        model_path: Path to trained model
        input_file: Optional path to WhatsApp export
    
    Returns:
        SentimentPredictor instance or results
    """
    predictor = SentimentPredictor(model_path)
    
    if input_file:
        results = predictor.predict_from_file(input_file)
        return results
    
    return predictor


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Inference Module")
    print("Usage: from inference import SentimentPredictor")
    print()
    print("# Load model")
    print("predictor = SentimentPredictor('path/to/model.pkl')")
    print()
    print("# Single prediction")
    print("sentiment = predictor.predict_single('This is amazing!')")
    print()
    print("# Batch prediction")
    print("results = predictor.predict_batch(['I love this!', 'This is bad'])")
    print()
    print("# WhatsApp export")
    print("df = predictor.predict_from_file('chat.txt', 'output.csv')")
