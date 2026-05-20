"""
Prediction Module for Fake News Detection
Load trained model and make predictions on new articles
"""

import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Union

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"


class FakeNewsPredictor:
    """Predict fake news using trained model"""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model
            vectorizer_path: Path to TF-IDF vectorizer
        """
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Default paths - use Logistic Regression model as it has predict_proba
        if model_path is None:
            self.model_path = MODELS_DIR / "logistic_regression_model.pkl"
        if vectorizer_path is None:
            self.vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded model from: {self.model_path}")
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Loaded vectorizer from: {self.vectorizer_path}")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found. {e}")
            print("Please train the model first using train_model.py")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for prediction
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text: str) -> Tuple[int, str]:
        """
        Predict if news is fake or real
        
        Args:
            text: News article text
            
        Returns:
            Tuple of (prediction, label)
            prediction: 0 for fake, 1 for real
            label: String label
        """
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        if len(processed_text) < 20:
            return None, "Text too short to analyze"
        
        # Transform using TF-IDF
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get label
        label = "REAL" if prediction == 1 else "FAKE"
        
        return prediction, label
    
    def predict_proba(self, text: str) -> Tuple[Dict[str, float], str]:
        """
        Predict with probability scores
        
        Args:
            text: News article text
            
        Returns:
            Tuple of (probabilities, predicted_label)
        """
        processed_text = self.preprocess_text(text)
        
        if len(processed_text) < 20:
            return {"FAKE": 0.5, "REAL": 0.5}, "Text too short"
        
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Get probabilities if available
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(text_tfidf)[0]
                probabilities = {
                    "FAKE": round(proba[0], 4),
                    "REAL": round(proba[1], 4)
                }
            else:
                # Use decision function
                decision = self.model.decision_function(text_tfidf)[0]
                # Normalize to pseudo-probabilities
                prob_fake = 1 / (1 + np.exp(decision))
                prob_real = 1 - prob_fake
                probabilities = {
                    "FAKE": round(prob_fake, 4),
                    "REAL": round(prob_real, 4)
                }
        except:
            probabilities = {"FAKE": 0.5, "REAL": 0.5}
        
        # Get predicted label
        prediction = self.model.predict(text_tfidf)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        
        return probabilities, label
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive analysis of text

        Args:
            text: News article text

        Returns:
            Dictionary with analysis results
        """
        prediction, label = self.predict(text)
        probabilities, _ = self.predict_proba(text)

        # Calculate confidence
        confidence = max(probabilities.values())

        # Text statistics
        word_count = len(text.split())
        char_count = len(text)

        result = {
            "prediction": label,
            "prediction_code": int(prediction) if prediction is not None else None,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "word_count": int(word_count),
            "char_count": int(char_count),
            "analysis": self._get_analysis_text(label, confidence)
        }

        return result
    
    def _get_analysis_text(self, label: str, confidence: float) -> str:
        """Generate analysis explanation"""
        conf_percent = confidence * 100
        
        if label == "REAL":
            if conf_percent > 80:
                return f"This article appears to be REAL with {conf_percent:.1f}% confidence. The writing style and content patterns are consistent with authentic news sources."
            else:
                return f"This article is likely REAL but with moderate confidence ({conf_percent:.1f}%). Consider additional verification."
        else:
            if conf_percent > 80:
                return f"This article appears to be FAKE with {conf_percent:.1f}% confidence. The content shows patterns typical of misinformation."
            else:
                return f"This article shows some indicators of being FAKE but with moderate confidence ({conf_percent:.1f}%). Exercise caution."


def predict_news(text: str) -> Dict:
    """
    Quick function to predict if news is fake or real
    
    Args:
        text: News article text
        
    Returns:
        Dictionary with prediction results
    """
    predictor = FakeNewsPredictor()
    return predictor.analyze_text(text)


def batch_predict(texts: list) -> list:
    """
    Predict for multiple texts
    
    Args:
        texts: List of news article texts
        
    Returns:
        List of prediction results
    """
    predictor = FakeNewsPredictor()
    results = []
    
    for text in texts:
        result = predictor.analyze_text(text)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test the predictor
    print("="*60)
    print("FAKE NEWS PREDICTION TEST")
    print("="*60)
    
    # Sample test texts
    test_articles = [
        # Real news examples
        "The government announced today new measures to improve the economy. The policy includes tax reforms and investments in infrastructure. Experts say this could boost growth by 2% next year.",
        
        # Fake news examples  
        "BREAKING: Secret document reveals that famous celebrities have been hiding a massive conspiracy. The truth has been suppressed by mainstream media. Share this before they delete it!",
        
        # Another real example
        "Stock markets around the world rose today following positive economic data. The technology sector led the gains, with major companies reporting better than expected quarterly earnings."
    ]
    
    try:
        predictor = FakeNewsPredictor()
        
        print("\nTesting predictions:")
        print("-"*60)
        
        for i, article in enumerate(test_articles, 1):
            print(f"\nArticle {i}:")
            print(f"Text: {article[:80]}...")
            
            result = predictor.analyze_text(article)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Probabilities: {result['probabilities']}")
            print(f"Analysis: {result['analysis']}")
            print("-"*60)
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first by running train_model.py")

