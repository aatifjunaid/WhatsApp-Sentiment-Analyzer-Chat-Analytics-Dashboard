"""Phase 3: Feature Extraction"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_SEQ_LENGTH

class FeatureExtractor:
    """Extract features from preprocessed text"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.embeddings = None
    
    def tfidf_vectorization(self, texts, max_features=5000, ngram_range=(1, 2), 
                           fit=True, vocabulary=None):
        """
        Create TF-IDF features
        
        Args:
            texts: List of text strings
            max_features: Maximum number of features
            ngram_range: Tuple of (min_n, max_n) for n-grams
            fit: Whether to fit the vectorizer
            vocabulary: Pre-existing vocabulary (for transform on new data)
        
        Returns:
            sparse matrix of TF-IDF features
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                lowercase=True,
                stop_words='english'
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
            print(f"TF-IDF features shape: {features.shape}")
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Set fit=True first.")
            features = self.tfidf_vectorizer.transform(texts)
        
        return features
    
    def get_tfidf_feature_names(self):
        """Get feature names from TF-IDF vectorizer"""
        if self.tfidf_vectorizer is None:
            return None
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def get_top_features(self, feature_weights, top_n=10):
        """Get top N features by weight"""
        top_indices = np.argsort(feature_weights)[-top_n:][::-1]
        feature_names = self.get_tfidf_feature_names()
        
        return [(feature_names[i], feature_weights[i]) for i in top_indices]
    
    def bert_embeddings(self, texts, model_name="distilbert-base-uncased"):
        """
        Generate embeddings using pre-trained transformer model
        
        Args:
            texts: List of text strings
            model_name: HuggingFace model identifier
        
        Returns:
            numpy array of embeddings
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            print("Please install transformers and torch: pip install transformers torch")
            return None
        
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        embeddings_list = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and encode
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LENGTH,
                    padding=True,
                    truncation=True
                )
                
                # Get embeddings
                outputs = model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        self.embeddings = embeddings
        print(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def word2vec_embeddings(self, texts, vector_size=100, window=5, min_count=2):
        """
        Generate embeddings using Word2Vec
        
        Args:
            texts: List of tokenized text (list of lists)
            vector_size: Dimension of embeddings
            window: Context window size
            min_count: Minimum word frequency
        
        Returns:
            numpy array of averaged embeddings per document
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("Please install gensim: pip install gensim")
            return None
        
        print("Training Word2Vec model...")
        tokenized_texts = [text.split() for text in texts]
        
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        
        # Average word vectors for each document
        embeddings = []
        for tokens in tokenized_texts:
            valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
            if valid_vectors:
                embeddings.append(np.mean(valid_vectors, axis=0))
            else:
                embeddings.append(np.zeros(vector_size))
        
        embeddings = np.array(embeddings)
        self.embeddings = embeddings
        print(f"Word2Vec embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def glove_embeddings(self, texts, vector_size=100):
        """
        Generate embeddings using GloVe vectors
        
        Prerequisites: Download GloVe vectors and provide path
        
        Args:
            texts: List of text strings
            vector_size: Dimension of embeddings (100, 200, 300)
        
        Returns:
            numpy array of averaged embeddings per document
        """
        print(f"Loading GloVe embeddings (dimension={vector_size})...")
        
        # This would require downloading GloVe vectors
        # For now, returning placeholder
        print("GloVe embeddings require pre-downloaded vector file.")
        print("Download from: https://nlp.stanford.edu/projects/glove/")
        
        return None


class FeatureStatistics:
    """Compute and display feature statistics"""
    
    @staticmethod
    def analyze_tfidf_features(tfidf_matrix, feature_names, top_n=20):
        """Analyze TF-IDF features"""
        # Calculate mean TF-IDF for each feature
        mean_tfidf = tfidf_matrix.mean(axis=0).A1  # Convert to 1D array
        
        # Get top features
        top_indices = np.argsort(mean_tfidf)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'feature': feature_names[idx],
                'mean_tfidf': mean_tfidf[idx],
                'occurrences': (tfidf_matrix[:, idx] > 0).sum()
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def embedding_statistics(embeddings):
        """Compute statistics on embeddings"""
        stats = {
            'shape': embeddings.shape,
            'mean': embeddings.mean(axis=0),
            'std': embeddings.std(axis=0),
            'min': embeddings.min(axis=0),
            'max': embeddings.max(axis=0),
        }
        return stats


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    sample_texts = [
        "I love this product, it's amazing!",
        "This is terrible and doesn't work",
        "It's okay, nothing special"
    ]
    
    # TF-IDF
    tfidf_features = extractor.tfidf_vectorization(sample_texts)
    print(f"TF-IDF shape: {tfidf_features.shape}")
