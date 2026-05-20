"""
Flask Web Application for Fake News Detection
Provides a web interface to check if news articles are fake or real
"""

from flask import Flask, render_template_string, request, jsonify
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import FakeNewsPredictor

app = Flask(__name__)

# Initialize predictor
predictor = None

def get_predictor():
    """Get or initialize the predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = FakeNewsPredictor()
        except FileNotFoundError:
            print("Warning: Model not found. Please train the model first.")
            predictor = None
    return predictor

# HTML Template
HTML_TEMPLATE = """
<!-- ADVANCED VERSION: Glassmorphism + Particles + Charts + History + Themes -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 150px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        .result.real {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .result.fake {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
        }
        
        .result-label {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .confidence {
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .probabilities {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }
        
        .prob-item {
            flex: 1;
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .prob-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .analysis {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.3);
            font-size: 1em;
            line-height: 1.6;
        }
        
        .error {
            background: #fee;
            color: #c00;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .loading {
            text-align: center;
            display: none;
            padding: 20px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Fake News Detector</h1>
            <p>AI-powered detection of real vs fake news articles</p>
        </div>
        
        <div class="content">
            <div class="input-group">
                <label for="newsText">📰 Paste your news article below:</label>
                <textarea id="newsText" placeholder="Enter the news article text you want to check..."></textarea>
            </div>
            
            <button class="btn" onclick="analyzeNews()">Analyze News</button>
            
            <div class="loading" id="loading">
                <p>🔄 Analyzing article...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <div class="result-label" id="resultLabel"></div>
                <div class="confidence" id="confidence"></div>
                <div class="probabilities">
                    <div class="prob-item">
                        <div>🤥 Fake</div>
                        <div class="prob-value" id="probFake"></div>
                    </div>
                    <div class="prob-item">
                        <div>✅ Real</div>
                        <div class="prob-value" id="probReal"></div>
                    </div>
                </div>
                <div class="analysis" id="analysis"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>Machine Learning Model | TF-IDF + Logistic Regression</p>
        </div>
    </div>
    
    <script>
        async function analyzeNews() {
            const text = document.getElementById('newsText').value.trim();
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            
            // Hide previous results
            resultDiv.classList.remove('show');
            errorDiv.style.display = 'none';
            
            if (!text) {
                errorDiv.textContent = 'Please enter some text to analyze!';
                errorDiv.style.display = 'block';
                return;
            }
            
            if (text.length < 50) {
                errorDiv.textContent = 'Please enter at least 50 characters for accurate analysis.';
                errorDiv.style.display = 'block';
                return;
            }
            
            // Show loading
            loadingDiv.style.display = 'block';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text }),
                });
                
                const data = await response.json();
                
                loadingDiv.style.display = 'none';
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                    return;
                }
                
                // Show result
                const isReal = data.prediction === 'REAL';
                resultDiv.className = 'result show ' + (isReal ? 'real' : 'fake');
                
                document.getElementById('resultLabel').textContent = 
                    (isReal ? '✅ REAL NEWS' : '🤥 FAKE NEWS');
                
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                
                document.getElementById('probFake').textContent = 
                    `${(data.probabilities.FAKE * 100).toFixed(1)}%`;
                
                document.getElementById('probReal').textContent = 
                    `${(data.probabilities.REAL * 100).toFixed(1)}%`;
                
                document.getElementById('analysis').textContent = data.analysis;
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                errorDiv.textContent = 'Error analyzing news. Please try again.';
                errorDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Render the home page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        predictor = get_predictor()
        
        if predictor is None:
            return jsonify({
                'error': 'Model not found. Please train the model first using train_model.py'
            }), 500
        
        result = predictor.analyze_text(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    predictor = get_predictor()
    return jsonify({
        'status': 'healthy' if predictor else 'model_not_loaded',
        'model_loaded': predictor is not None
    })


if __name__ == '__main__':
    print("="*60)
    print("Starting Fake News Detection Web App...")
    print("="*60)
    print("\nMake sure you have trained the model first!")
    print("Run: python src/train_model.py")
    print("\nThen open your browser to: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

