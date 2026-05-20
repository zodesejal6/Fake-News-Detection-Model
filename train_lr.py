"""
Train and save the Logistic Regression model specifically
This ensures we have a model with predict_proba for probability predictions
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import NewsDataLoader

# Paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train():
    print("="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    loader = NewsDataLoader()
    X, y = loader.prepare_data()
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create vectorizer
    print("\n3. Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train model
    print("\n4. Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    print("\n6. Saving model...")
    
    # Save model
    model_path = MODELS_DIR / "logistic_regression_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, vectorizer

if __name__ == "__main__":
    train()

