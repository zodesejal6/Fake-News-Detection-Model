"""
Command Line Interface for Fake News Detection
Allows users to check if news articles are fake or real from the terminal
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import FakeNewsPredictor
from train_model import train_and_save


def print_result(result: dict):
    """Pretty print the prediction result"""
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULT")
    print("="*60)
    
    # Prediction
    prediction = result['prediction']
    is_real = prediction == "REAL"
    
    if is_real:
        print(f"✅ Prediction: REAL NEWS")
    else:
        print(f"🤥 Prediction: FAKE NEWS")
    
    # Confidence
    confidence = result['confidence'] * 100
    print(f"📈 Confidence: {confidence:.1f}%")
    
    # Probabilities
    print(f"\n📉 Probability Breakdown:")
    print(f"   Fake: {result['probabilities']['FAKE']*100:.1f}%")
    print(f"   Real: {result['probabilities']['REAL']*100:.1f}%")
    
    # Analysis
    print(f"\n📝 Analysis:")
    print(f"   {result['analysis']}")
    
    # Text stats
    print(f"\n📏 Text Statistics:")
    print(f"   Word count: {result['word_count']}")
    print(f"   Character count: {result['char_count']}")
    
    print("="*60)


def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "="*60)
    print("🔍 FAKE NEWS DETECTOR - Interactive Mode")
    print("="*60)
    print("\nEnter news articles to check if they're real or fake.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        predictor = FakeNewsPredictor()
    except FileNotFoundError:
        print("❌ Error: Model not found!")
        print("Please train the model first by running: python main.py --train")
        return
    
    while True:
        try:
            print("\n" + "-"*60)
            text = input("📰 Paste your news article (or type 'quit' to exit):\n> ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text.strip():
                print("⚠️ Please enter some text!")
                continue
            
            if len(text.strip()) < 50:
                print("⚠️ Please enter at least 50 characters for accurate analysis.")
                continue
            
            result = predictor.analyze_text(text)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def check_text(text: str):
    """Check a specific text"""
    try:
        predictor = FakeNewsPredictor()
    except FileNotFoundError:
        print("❌ Error: Model not found!")
        print("Please train the model first by running: python main.py --train")
        return
    
    result = predictor.analyze_text(text)
    print_result(result)


def check_file(file_path: str):
    """Check text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    check_text(text)


def train_model():
    """Train the model"""
    print("\n" + "="*60)
    print("🎓 Starting Model Training...")
    print("="*60)
    
    trainer = train_and_save()
    
    print("\n✅ Training complete! You can now use the model for predictions.")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Fake News Detection CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py --train            # Train the model
  python main.py --text "news..."   # Check specific text
  python main.py --file article.txt # Check text from file
  python main.py --demo             # Run demo predictions
        """
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--text', type=str,
                        help='Text to check for fake news')
    parser.add_argument('--file', type=str,
                        help='File containing text to check')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo predictions')
    
    args = parser.parse_args()
    
    # Train mode
    if args.train:
        train_model()
        return
    
    # File mode
    if args.file:
        check_file(args.file)
        return
    
    # Text mode
    if args.text:
        check_text(args.text)
        return
    
    # Demo mode
    if args.demo:
        print("\n" + "="*60)
        print("🎬 Running Demo Predictions")
        print("="*60)
        
        demo_texts = [
            ("Real News Example 1", 
             "The federal reserve announced today that interest rates will remain unchanged. "
             "The decision comes after careful consideration of economic indicators including "
             "inflation, employment rates, and global market conditions."),
            
            ("Fake News Example 1",
             "BREAKING: Scientists discover miracle cure that pharmaceutical companies don't "
             "want you to know about! This simple remedy can cure all diseases but big pharma "
             "is hiding it from the public. Share this before they delete it!"),
            
            ("Real News Example 2",
             "Technology companies reported strong quarterly earnings this week, with major "
             "firms exceeding analyst expectations. The tech sector led market gains amid "
             "continued demand for digital services and products."),
            
            ("Fake News Example 2",
             "URGENT WARNING: A famous celebrity has predicted that the world will end "
             "in just 3 days! According to their calculations, a massive disaster will "
             "strike the entire planet. Everyone must prepare immediately!"),
        ]
        
        try:
            predictor = FakeNewsPredictor()
            
            for title, text in demo_texts:
                print(f"\n{'='*60}")
                print(f"Test: {title}")
                print(f"{'='*60}")
                result = predictor.analyze_text(text)
                print_result(result)
                
        except FileNotFoundError:
            print("❌ Error: Model not found!")
            print("Please train the model first by running: python main.py --train")
        return
    
    # Default: interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()

