import joblib
import os
from pathlib import Path

# ============================================================================
# SENTIMENT PREDICTOR - Use Trained Models Locally
# ============================================================================

class SentimentPredictor:
    """Use trained models to predict sentiment on any text"""
    
    def __init__(self, model_folder='C:\\Users\\Lenovo\\Downloads'):
        """
        Load trained models from your Downloads folder
        
        Args:
            model_folder: Path to folder with downloaded models
        """
        print("Loading models...")
        
        try:
            # Load the three components
            self.lr_model = joblib.load(os.path.join(model_folder, 'lr_model.pkl'))
            self.svm_model = joblib.load(os.path.join(model_folder, 'svm_model.pkl'))
            self.tfidf = joblib.load(os.path.join(model_folder, 'tfidf_vectorizer.pkl'))
            
            print("✅ Models loaded successfully!\n")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find model files in {model_folder}")
            print(f"Make sure these files are in that folder:")
            print(f"  - lr_model.pkl")
            print(f"  - svm_model.pkl")
            print(f"  - tfidf_vectorizer.pkl")
            raise
    
    def predict(self, text, model='logistic_regression'):
        """
        Predict sentiment of text
        
        Args:
            text: Text to analyze
            model: 'logistic_regression' or 'svm'
            
        Returns:
            Dictionary with sentiment and confidence
        """
        # Convert text to TF-IDF features
        text_vectorized = self.tfidf.transform([text])
        
        # Choose model
        if model == 'logistic_regression':
            prediction = self.lr_model.predict(text_vectorized)[0]
            probabilities = self.lr_model.predict_proba(text_vectorized)[0]
        else:
            prediction = self.svm_model.predict(text_vectorized)[0]
            probabilities = self.svm_model.predict_proba(text_vectorized)[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        return {
            'text': text,
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': {
                'positive': probabilities[0] if 'positive' in self.lr_model.classes_ else 0,
                'neutral': probabilities[1] if 'neutral' in self.lr_model.classes_ else 0,
                'negative': probabilities[2] if 'negative' in self.lr_model.classes_ else 0,
            }
        }
    
    def predict_batch(self, texts, model='logistic_regression'):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            model: 'logistic_regression' or 'svm'
            
        Returns:
            List of sentiment predictions
        """
        results = []
        for text in texts:
            result = self.predict(text, model)
            results.append(result)
        return results
    
    def predict_from_file(self, filepath):
        """
        Predict sentiment for each line in a text file
        
        Args:
            filepath: Path to .txt file
            
        Returns:
            List of predictions
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = self.predict_batch([line.strip() for line in lines])
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    print("=" * 70)
    print("🎯 SENTIMENT PREDICTION - LOCAL USAGE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Single Text Prediction
    print("\n1️⃣ Single Text Prediction:\n")
    
    text = "I absolutely LOVE this airline! Best service ever! 😍"
    result = predictor.predict(text)
    
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    # Example 2: Multiple Text Predictions
    print("\n\n2️⃣ Multiple Texts:\n")
    
    test_texts = [
        "This is amazing! I love it!",
        "It's okay, nothing special",
        "This is terrible, worst ever",
        "Pretty good, would recommend",
        "Absolutely horrible experience 😡"
    ]
    
    results = predictor.predict_batch(test_texts)
    
    for i, result in enumerate(results, 1):
        text = result['text'][:40] + "..." if len(result['text']) > 40 else result['text']
        sentiment = result['sentiment'].upper()
        confidence = result['confidence']
        
        print(f"{i}. [{sentiment:8}] {text:45} (Confidence: {confidence:.0%})")
    
    # Example 3: Using SVM Model
    print("\n\n3️⃣ Using SVM Model (Alternative):\n")
    
    text = "Not bad, but could be better"
    result_lr = predictor.predict(text, model='logistic_regression')
    result_svm = predictor.predict(text, model='svm')
    
    print(f"Text: {text}")
    print(f"Logistic Regression: {result_lr['sentiment'].upper()} ({result_lr['confidence']:.1%})")
    print(f"SVM:                 {result_svm['sentiment'].upper()} ({result_svm['confidence']:.1%})")
    
    # Example 4: Analyze File
    print("\n\n4️⃣ Analyze Text File (Optional):\n")
    
    # You can create a text file with multiple lines and analyze it:
    example_file = "example_texts.txt"
    
    if os.path.exists(example_file):
        print(f"Analyzing {example_file}...")
        results = predictor.predict_from_file(example_file)
        print(f"Found {len(results)} lines")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result['sentiment'].upper()}")
    
    print("\n" + "=" * 70)
    print("✅ Ready to use! See examples above")
    print("=" * 70)
