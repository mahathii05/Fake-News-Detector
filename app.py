import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ------------------ LOAD NLP TOOLS ------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ------------------ UI ------------------
st.title("📰 Fake News Detection Web App")
st.write(
    "Paste a news article below and the system will predict "
    "whether the news is **Fake** or **Real** using AI."
)

news_input = st.text_area(
    "Enter News Article",
    height=220,
    placeholder="Paste the news content here..."
)

# ------------------ PREDICTION ------------------
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        cleaned_text = clean_text(news_input)
        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]
        confidence = max(probability) * 100

        st.markdown("---")

        if prediction == 1:
            st.success(f"Hey! This news is REAL\n\nConfidence: **{confidence:.2f}%**")
        else:
            st.error(f" Oops! This news is FAKE\n\nConfidence: **{confidence:.2f}%**")


