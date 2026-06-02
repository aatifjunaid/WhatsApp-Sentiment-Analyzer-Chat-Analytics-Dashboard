"""Phase 4: Model Training & Selection"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RANDOM_STATE, DEVICE


class ClassicalMLModels:
    """Classical machine learning models for sentiment classification"""
    
    def __init__(self):
        self.logistic_regression = None
        self.svm = None
        self.models = {}
    
    def logistic_regression_model(self, X_train, y_train, C=1.0, max_iter=1000):
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features (sparse or dense matrix)
            y_train: Training labels
            C: Regularization strength
            max_iter: Maximum iterations
        
        Returns:
            Trained model
        """
        print("Training Logistic Regression...")
        self.logistic_regression = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1
        )
        self.logistic_regression.fit(X_train, y_train)
        self.models['logistic_regression'] = self.logistic_regression
        print("Logistic Regression trained")
        return self.logistic_regression
    
    def svm_model(self, X_train, y_train, kernel='rbf', C=1.0):
        """
        Train Support Vector Machine model
        
        Args:
            X_train: Training features
            y_train: Training labels
            kernel: SVM kernel type ('rbf', 'linear', 'poly')
            C: Regularization strength
        
        Returns:
            Trained model
        """
        print(f"Training SVM with kernel={kernel}...")
        self.svm = SVC(
            kernel=kernel,
            C=C,
            random_state=RANDOM_STATE,
            probability=True,
            n_jobs=-1
        )
        self.svm.fit(X_train, y_train)
        self.models['svm'] = self.svm
        print("SVM trained")
        return self.svm
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='logistic_regression', 
                             param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to tune
            param_grid: Parameter grid for GridSearchCV
            cv: Number of cross-validation folds
        
        Returns:
            Best model and best parameters
        """
        if model_type == 'logistic_regression':
            if param_grid is None:
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        
        elif model_type == 'svm':
            if param_grid is None:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            base_model = SVC(random_state=RANDOM_STATE, probability=True)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Tuning {model_type} hyperparameters...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.models[model_type] = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_
    
    def save_model(self, model, model_name):
        """Save model to disk"""
        save_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_name):
        """Load model from disk"""
        load_path = MODELS_DIR / f"{model_name}.pkl"
        model = joblib.load(load_path)
        print(f"Model loaded from {load_path}")
        return model


class DeepLearningModels:
    """Deep Learning models for sentiment classification"""
    
    def __init__(self):
        self.lstm_model = None
        self.bert_model = None
    
    def build_lstm_model(self, vocab_size, embedding_dim=100, max_length=128, num_classes=3):
        """
        Build LSTM model for sentiment classification
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            num_classes: Number of sentiment classes
        
        Returns:
            Compiled PyTorch or TensorFlow model
        """
        print("Building LSTM model...")
        
        try:
            import torch
            import torch.nn as nn
            
            class LSTMModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
                    super(LSTMModel, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                    self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
                    self.dropout = nn.Dropout(0.5)
                    self.fc = nn.Linear(256, num_classes)  # 256 = 2 * 128 (bidirectional)
                
                def forward(self, x):
                    x = self.embedding(x)
                    lstm_out, _ = self.lstm(x)
                    x = lstm_out[:, -1, :]  # Take last output
                    x = self.dropout(x)
                    x = self.fc(x)
                    return x
            
            self.lstm_model = LSTMModel(vocab_size, embedding_dim, max_length, num_classes)
            print("LSTM model built successfully")
            return self.lstm_model
        
        except ImportError:
            print("PyTorch not installed. Please install: pip install torch")
            return None
    
    def fine_tune_distilbert(self, train_dataloader, val_dataloader, 
                            num_epochs=3, learning_rate=2e-5):
        """
        Fine-tune DistilBERT for sentiment classification
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Fine-tuned model
        """
        print("Fine-tuning DistilBERT...")
        
        try:
            from transformers import DistilBertForSequenceClassification, AdamW
            import torch
        except ImportError:
            print("transformers or torch not installed.")
            return None
        
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3
        )
        model.to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_loss:.4f}")
        
        self.bert_model = model
        return model


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(y_true, y_pred, labels=None):
        """
        Compute evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names (for reporting)
        
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, label_names=None):
        """Print metrics in readable format"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
        print(f"Precision (macro):     {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):        {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):      {metrics['f1_macro']:.4f}")
        print(f"Precision (weighted):  {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted):     {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (weighted):   {metrics['f1_weighted']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])


if __name__ == "__main__":
    # Example usage would go here
    pass
