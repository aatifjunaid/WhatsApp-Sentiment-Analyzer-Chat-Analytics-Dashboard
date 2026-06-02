"""Phase 5: Model Evaluation and Visualization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENTIMENT_LABELS


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.results = {}
    
    def evaluate_predictions(self, y_true, y_pred, y_proba=None):
        """Generate full evaluation report"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                                                          target_names=SENTIMENT_LABELS,
                                                          output_dict=True)
        }
        
        self.results = results
        return results
    
    def print_report(self, results=None):
        """Print formatted evaluation report"""
        if results is None:
            results = self.results
        
        print("\n" + "="*70)
        print(f"EVALUATION REPORT - {self.model_name}")
        print("="*70)
        
        print("\nOVERALL METRICS:")
        print(f"  Accuracy:              {results['accuracy']:.4f}")
        print(f"  Precision (macro):     {results['precision_macro']:.4f}")
        print(f"  Recall (macro):        {results['recall_macro']:.4f}")
        print(f"  F1-Score (macro):      {results['f1_macro']:.4f}")
        print(f"  Precision (weighted):  {results['precision_weighted']:.4f}")
        print(f"  Recall (weighted):     {results['recall_weighted']:.4f}")
        print(f"  F1-Score (weighted):   {results['f1_weighted']:.4f}")
        
        print("\nCONFUSION MATRIX:")
        cm = np.array(results['confusion_matrix'])
        print(cm)
        
        print("\nPER-CLASS METRICS:")
        for label, metrics in results['classification_report'].items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"\n  {SENTIMENT_LABELS[int(label)].upper()}:")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall:    {metrics.get('recall', 0):.4f}")
                print(f"    F1-Score:  {metrics.get('f1-score', 0):.4f}")
        
        print("\n" + "="*70)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=SENTIMENT_LABELS,
                   yticklabels=SENTIMENT_LABELS,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_metrics_comparison(self, results_dict, save_path=None):
        """Compare metrics across multiple models"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        data = {metric: [results_dict[model].get(metric, 0) for model in models] 
                for metric in metrics}
        
        df = pd.DataFrame(data, index=models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax)
        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        plt.show()
    
    def save_results(self, filepath):
        """Save results to JSON"""
        results_to_save = self.results.copy()
        # Convert numpy arrays to lists for JSON serialization
        if 'confusion_matrix' in results_to_save:
            results_to_save['confusion_matrix'] = np.array(results_to_save['confusion_matrix']).tolist()
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"Results saved to {filepath}")


class CrossValidationEvaluator:
    """Cross-validation evaluation"""
    
    @staticmethod
    def evaluate_cv(model, X, y, cv=5, scoring='f1_weighted'):
        """Perform cross-validation"""
        from sklearn.model_selection import cross_val_score, cross_validate
        
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"\nCross-Validation Results:")
        print(f"  Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f}")
        print(f"  Std Dev: {cv_scores.std():.4f}")
        print(f"  Min: {cv_scores.min():.4f}")
        print(f"  Max: {cv_scores.max():.4f}")
        
        return cv_scores


if __name__ == "__main__":
    # Example usage
    from sklearn.metrics import confusion_matrix
    
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
    
    evaluator = ModelEvaluator("Example Model")
    results = evaluator.evaluate_predictions(y_true, y_pred)
    evaluator.print_report()
