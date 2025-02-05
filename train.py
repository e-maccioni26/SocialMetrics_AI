from db import get_db_connection
from model import train_model
import pandas as pd

def fetch_tweets_data():
    conn = get_db_connection()
    query = "SELECT text, positive, negative FROM tweets"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def main():
    df = fetch_tweets_data()
    if df.empty:
        print("Aucune donnée disponible pour l'entraînement.")
        return
    texts = df['text'].tolist()
    positive_labels = df['positive'].tolist()
    negative_labels = df['negative'].tolist()
    train_model(texts, positive_labels, negative_labels)
    print("Modèle entraîné et sauvegardé avec succès.")

if __name__ == '__main__':
    main()
