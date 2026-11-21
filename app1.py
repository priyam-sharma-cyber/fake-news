
import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Indian Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("model_india.pkl")
    vectorizer = joblib.load("vectorizer_india.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Application UI
st.title("🇮🇳 Indian Fake News Detection")
st.subheader("AI Powered News Verification System")

st.write(
    "Enter an Indian news article below "
    "to classify it as Fake or Real."
)

news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste your news article here..."
)

if st.button("Check News", use_container_width=True):

    if not news.strip():
        st.warning("Please enter a news article.")

    else:
        cleaned = clean_text(news)

        if not cleaned:
            st.warning("Please enter valid English news text.")

        else:
            features = vectorizer.transform([cleaned])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            confidence = max(probabilities) * 100

            if prediction == 1:
                st.success("✅ Prediction: REAL NEWS")
            else:
                st.error("🚨 Prediction: FAKE NEWS")

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "This is a machine-learning prediction, "
                "not independent verification of the news."
            )

st.divider()

st.caption(
    "Indian Fake News Detection | "
    "Logistic Regression + TF-IDF"
)
