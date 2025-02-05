# config.py
import os

# Configuration de la base de données MySQL
DB_CONFIG = {
    'host': '127.0.0.1',         # Si l'API est lancée sur la machine hôte
    'user': 'sentiment_user',
    'password': 'sentiment_password',
    'database': 'sentiment_db'
}

# Chemin de sauvegarde du modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
