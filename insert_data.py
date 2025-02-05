from db import get_db_connection

# Exemple de données 
data = [
    ("J'adore ce produit, il est incroyable !", 1, 0),
    ("Ce service est excellent et très réactif.", 1, 0),
    ("Quelle expérience fantastique, je recommande vivement !", 1, 0),
    ("Je déteste ce service, très décevant.", 0, 1),
    ("Le produit est de très mauvaise qualité, je ne l'achèterai plus.", 0, 1),
    ("Expérience horrible, je suis très mécontent.", 0, 1)
]

def insert_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)"
    cursor.executemany(query, data)
    conn.commit()
    cursor.close()
    conn.close()
    print("Données insérées avec succès.")

if __name__ == '__main__':
    insert_data()
