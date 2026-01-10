import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

nltk.download('punkt')
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


def summarize_text(text, n_sentences=3):
    sentences = sent_tokenize(text)
    
    if len(sentences) <= n_sentences:
        return text

    tfidf = vectorizer.transform(sentences)

    sentence_scores = np.sum(tfidf.toarray(), axis=1)
    top_indices = sentence_scores.argsort()[-n_sentences:]

    summary = [sentences[i] for i in sorted(top_indices)]
    return " ".join(summary)
