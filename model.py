import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import os
from config import MODEL_PATH

def train_model(texts, positive_labels, negative_labels):
    model_positive = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model_positive.fit(texts, positive_labels)
    
    model_negative = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model_negative.fit(texts, negative_labels)
    
    joblib.dump((model_positive, model_negative), MODEL_PATH)
    return model_positive, model_negative

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model_positive, model_negative = joblib.load(MODEL_PATH)
    return model_positive, model_negative

def predict_sentiment(tweets):
    models = load_model()
    if models is None:
        raise Exception("Le modèle n'est pas encore entraîné.")
    model_positive, model_negative = models
    scores = {}
    for tweet in tweets:
        # Prédire la probabilité pour la classe positive et négative
        p_positive = model_positive.predict_proba([tweet])[0][1]  # probabilité pour la classe 1
        p_negative = model_negative.predict_proba([tweet])[0][1]
        score = p_positive - p_negative  # Score dans [-1, 1]
        scores[tweet] = score
    return scores

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return cm, report
