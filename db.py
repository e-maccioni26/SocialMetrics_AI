# db.py
import mysql.connector
from config import DB_CONFIG

def get_db_connection():
    connection = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    return connection

def create_tweets_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS tweets (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT NOT NULL,
        positive TINYINT(1) NOT NULL,
        negative TINYINT(1) NOT NULL
    )
    """
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    create_tweets_table()
    print("La table 'tweets' a été créée (si elle n'existait pas déjà).")
