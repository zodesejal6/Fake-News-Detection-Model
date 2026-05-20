"""
Data Loader and Preprocessing Module for Fake News Detection
Handles loading, cleaning, and preparing news data for machine learning
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

class NewsDataLoader:
    """Load and preprocess news data for fake news detection"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the dataset CSV file. If None, uses default location
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = DATA_DIR / "news_dataset.csv"
        
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the news dataset
        
        Returns:
            DataFrame with news articles and labels
        """
        print(f"Loading data from: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"Loaded {len(self.df)} articles")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def get_basic_stats(self) -> dict:
        """Get basic statistics about the dataset"""
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_articles': len(self.df),
            'columns': list(self.df.columns),
            'label_distribution': self.df['label'].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return stats
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
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
        
        # Remove special characters and numbers (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, text_column: str = 'text', label_column: str = 'label') -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for model training
        
        Args:
            text_column: Name of the column containing news text
            label_column: Name of the column containing labels
            
        Returns:
            Tuple of (X, y) - features and labels
        """
        if self.df is None:
            self.load_data()
        
        # Check if required columns exist
        if text_column not in self.df.columns:
            # Try alternative column names
            if 'title' in self.df.columns:
                print(f"Using 'title' column as text")
                text_column = 'title'
            else:
                raise ValueError(f"Text column '{text_column}' not found. Available: {list(self.df.columns)}")
        
        if label_column not in self.df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {list(self.df.columns)}")
        
        # Fill missing values
        self.df[text_column] = self.df[text_column].fillna('')
        
        # Preprocess text
        print("Preprocessing text data...")
        self.df['processed_text'] = self.df[text_column].apply(self.preprocess_text)
        
        # Remove empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 10]
        
        X = self.df['processed_text']
        y = self.df[label_column]
        
        print(f"Prepared {len(X)} samples")
        print(f"Label distribution: True={sum(y==1)}, Fake={sum(y==0)}")
        
        return X, y
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Get train/test split
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = self.prepare_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


def quick_load() -> Tuple[pd.Series, pd.Series]:
    """
    Quick function to load and prepare data
    
    Returns:
        Tuple of (X, y)
    """
    loader = NewsDataLoader()
    return loader.prepare_data()


if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    print("=" * 50)
    
    try:
        loader = NewsDataLoader()
        df = loader.load_data()
        print("\nDataset sample:")
        print(df.head(2))
        
        print("\nBasic statistics:")
        stats = loader.get_basic_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nPreparing data...")
        X, y = loader.prepare_data()
        print(f"Data shape: X={len(X)}, y={len(y)}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run download_dataset.py first to create the dataset!")

