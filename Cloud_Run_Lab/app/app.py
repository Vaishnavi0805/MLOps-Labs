from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Customer Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/analyze": "POST - Analyze sentiment of text"
        },
        "author": "Vaishnavi Sarmalkar"
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of customer review
    
    Request body:
    {
        "text": "This product is amazing!"
    }
    
    Response:
    {
        "text": "This product is amazing!",
        "sentiment": "positive",
        "polarity": 0.85,
        "subjectivity": 0.75
    }
    """
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Return results
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "interpretation": {
                "polarity": "Ranges from -1 (negative) to 1 (positive)",
                "subjectivity": "Ranges from 0 (objective) to 1 (subjective)"
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)