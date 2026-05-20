"""
Model Training Module for Fake News Detection
Trains multiple classifiers and evaluates their performance
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Import data loader
from data_loader import NewsDataLoader


class FakeNewsTrainer:
    """Train and evaluate fake news detection models"""
    
    def __init__(self):
        """Initialize the trainer"""
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.best_model_name = None
        
    def create_vectorizer(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        """
        Create TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
            
        Returns:
            TfidfVectorizer instance
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        return self.vectorizer
    
    def initialize_models(self):
        """Initialize models to train"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            ),
            'Passive Aggressive': PassiveAggressiveClassifier(
                max_iter=1000,
                random_state=42,
                C=0.1
            ),
            'Multinomial NB': MultinomialNB(alpha=0.1),
            'Linear SVC': LinearSVC(
                random_state=42,
                max_iter=2000,
                C=0.5
            )
        }
        print(f"Initialized {len(self.models)} models")
        
    def train_all_models(self, X_train, y_train):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*60)
        print("Training Models...")
        print("="*60)
        
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.models = trained_models
        print("\nAll models trained!")
        
    def evaluate_model(self, model, X_test, y_test, model_name: str = ""):
        """
        Evaluate a model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Try to get probability scores for ROC-AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = None
        
        # Print evaluation
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
        
        print(f"Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of results for all models
        """
        results = {}
        
        print("\n" + "="*60)
        print("Evaluating Models...")
        print("="*60)
        
        for name, model in self.models.items():
            results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        # Find best model
        best_accuracy = 0
        for name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\n{'='*60}")
        print(f"Best Model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        print(f"{'='*60}")
        
        return results
    
    def save_model(self, model_name: str = None):
        """
        Save the trained model and vectorizer
        
        Args:
            model_name: Name of the model to save. If None, saves the best model
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
            
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Save model
        model_path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save vectorizer
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        return model_path, vectorizer_path
    
    def load_model(self, model_name: str = None):
        """
        Load a trained model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        if model_name is None:
            model_name = self.best_model_name
            
        model_path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}_model.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        print(f"Loaded model from: {model_path}")
        
        return model


def train_and_save():
    """Main function to train and save the model"""
    print("="*60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    loader = NewsDataLoader()
    
    try:
        X, y = loader.prepare_data()
    except FileNotFoundError:
        print("Dataset not found. Please run download_dataset.py first!")
        return
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create vectorizer and transform data
    print("\n3. Creating TF-IDF features...")
    trainer = FakeNewsTrainer()
    trainer.create_vectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = trainer.vectorizer.fit_transform(X_train)
    X_test_tfidf = trainer.vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Initialize and train models
    print("\n4. Training models...")
    trainer.initialize_models()
    trainer.train_all_models(X_train_tfidf, y_train)
    
    # Evaluate models
    print("\n5. Evaluating models...")
    results = trainer.evaluate_all_models(X_test_tfidf, y_test)
    
    # Save best model
    print("\n6. Saving model...")
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    train_and_save()

