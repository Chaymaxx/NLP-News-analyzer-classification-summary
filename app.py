import streamlit as st
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from summarizer import summarize_text
from preprocessing import normalize_arabic

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(page_title="NLP News Analyzer", layout="centered")

# --------------------------------------------------
# Sélection de la langue
# --------------------------------------------------
language = st.selectbox(
    "Choose language / اختر اللغة",
    ("English", "Arabic")
)

# --------------------------------------------------
# Textes UI selon la langue
# --------------------------------------------------
if language == "English":
    st.title("NLP News Analyzer")
    st.write("Classification and automatic summarization of news articles")
    article_label = "Paste your article here"
    classify_btn = "Classify Article"
    summary_btn = "Generate Summary"
    model_label = "Choose classification model"
    summary_label = "Choose summary length"
else:
    st.title("محلل المقالات الإخبارية")
    st.write("تصنيف وتلخيص المقالات الإخبارية باستخدام معالجة اللغة الطبيعية")
    article_label = "الصق نص المقال هنا"
    classify_btn = "تصنيف المقال"
    summary_btn = "إنشاء الملخص"
    model_label = "اختر نموذج التصنيف"
    summary_label = "اختر طول الملخص (عدد الجمل)"

# --------------------------------------------------
# Choix du modèle
# --------------------------------------------------
model_choice = st.selectbox(
    model_label,
    ("Naive Bayes (default)", "GRU")
)

# --------------------------------------------------
# Chargement des modèles
# --------------------------------------------------
@st.cache_resource
def load_nb_model(lang):
    if lang == "English":
        model = joblib.load("model/naive_bayes_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    else:
        model = joblib.load("model/naive_bayes_model_arabic.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer_arabic.pkl")
    return model, vectorizer


@st.cache_resource
def load_gru_model(lang):
    if lang == "English":
        model = load_model("model/gru_news_model.h5")
        with open("model/tokenizer_gru.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("model/label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
    else:
        model = load_model("model/gru_news_model_arabic.h5")
        with open("model/tokenizer_gru_arabic.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("model/label_encoder_arabic.pkl", "rb") as f:
            encoder = pickle.load(f)
    return model, tokenizer, encoder

# --------------------------------------------------
# Zone de texte
# --------------------------------------------------
article = st.text_area(article_label, height=250)

# --------------------------------------------------
# Classification
# --------------------------------------------------
if st.button(classify_btn):
    if article.strip():

        processed_article = article
        if language == "Arabic":
            processed_article = normalize_arabic(article)

        if model_choice == "Naive Bayes (default)":
            model, vectorizer = load_nb_model(language)
            X = vectorizer.transform([processed_article])
            prediction = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X))

        else:  # GRU
            model, tokenizer, encoder = load_gru_model(language)
            seq = tokenizer.texts_to_sequences([processed_article])

            padded = pad_sequences(
                seq,
                maxlen=300,
                padding='post',
                truncating='post'
            )

            probs = model.predict(padded)
            prediction = encoder.inverse_transform([np.argmax(probs)])[0]
            confidence = np.max(probs)

        if language == "English":
            st.success(f"Category: {prediction}")
            st.info(f"Confidence: {confidence*100:.2f}%")
        else:
            st.success(f"الفئة المتوقعة: {prediction}")
            st.info(f"نسبة الثقة: {confidence*100:.2f}%")

    else:
        st.warning("Please enter an article." if language == "English" else "يرجى إدخال نص المقال")

# --------------------------------------------------
# Résumé
# --------------------------------------------------
st.divider()

summary_length = st.slider(summary_label, 2, 7, 3)

if st.button(summary_btn):
    if article.strip():
        summary = summarize_text(article, summary_length)
        st.subheader("Summary" if language == "English" else "الملخص")
        st.write(summary)
    else:
        st.warning("Please enter an article." if language == "English" else "يرجى إدخال نص المقال")
