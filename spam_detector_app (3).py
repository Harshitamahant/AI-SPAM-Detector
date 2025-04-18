
import streamlit as st
import pandas as pd
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setup
st.set_page_config(page_title="Spam Email Detector", layout="centered")
download('stopwords')
stop_words = set(stopwords.words('english'))

# Title
st.title("📧 Spam Email Detector")
st.markdown("Enter an email message to check if it's **SPAM** or **HAM**.")

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['cleaned'] = df['message'].apply(preprocess_text)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label_num']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_model()

# User input
user_input = st.text_area("✉️ Email Content", height=150, placeholder="Type or paste your email here...")

if st.button("🔍 Predict"):
    if user_input.strip():
        cleaned = preprocess_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)

        if prediction[0] == 1:
            st.error("🔒 This is a **SPAM** message!")
        else:
            st.success("✅ This is a **HAM** (not spam) message.")
    else:
        st.warning("Please enter an email message to analyze.")
