import os

DB_CONFIG = {
    'host': '127.0.0.1',         
    'user': 'sentiment_user',
    'password': 'sentiment_password',
    'database': 'sentiment_db'
}

# Chemin de sauvegarde du modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
