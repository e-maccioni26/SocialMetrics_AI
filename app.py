from flask import Flask, request, jsonify, render_template
from model import predict_sentiment
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze_tweet():
    try:
        tweet = request.form.get('tweet', '').strip()
        if not tweet:
            return render_template('index.html', error="Veuillez saisir un tweet.", sentiment=None)

        scores = predict_sentiment([tweet])
        sentiment = scores[tweet]

        return render_template('index.html', sentiment=sentiment)
    except Exception as e:
        traceback.print_exc()
        return render_template('index.html', error=str(e), sentiment=None)

@app.route('/analyze', methods=['POST'])
def analyze_sentiments_api():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()
        if not data or 'tweets' not in data:
            return jsonify({"error": "Entrée invalide. Veuillez fournir la clé 'tweets' contenant une liste de tweets."}), 400
        tweets = data['tweets']
        if not isinstance(tweets, list) or not all(isinstance(t, str) for t in tweets):
            return jsonify({"error": "La clé 'tweets' doit être une liste de chaînes de caractères."}), 400
        if len(tweets) == 0:
            return jsonify({"error": "La liste de tweets est vide."}), 400

        scores = predict_sentiment(tweets)
        return jsonify(scores), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
