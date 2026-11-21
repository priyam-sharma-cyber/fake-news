
import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Indian Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#081226,#101d3b,#172554);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827,#172554);
}
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(100deg,#7c3aed,#2563eb,#0891b2);
    margin-bottom: 25px;
    box-shadow: 0 8px 25px #0005;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #e0f2fe;
}
.card {
    background: #1e293b;
    border: 1px solid #475569;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
div.stButton > button {
    background: linear-gradient(90deg,#7c3aed,#2563eb);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    min-height: 45px;
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #93c5fd;
}
h2, h3 {
    color: #93c5fd;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("model_india.pkl")
    vectorizer = joblib.load("vectorizer_india.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Model loading error!")
    st.exception(e)
    st.stop()

# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("# 📰 NewsGuard India")
    st.caption("AI Based News Detection System")

    st.markdown("---")

    st.markdown("### 🧠 Model Details")
    st.write("Algorithm: Logistic Regression")
    st.write("Vectorizer: TF-IDF")
    st.write("Classification: Binary")
    st.write("Fake Label: 0")
    st.write("Real Label: 1")

    st.markdown("---")

    st.markdown("### 📌 Instructions")
    st.write("1. Enter news article.")
    st.write("2. Click Analyse News.")
    st.write("3. Check prediction.")
    st.write("4. View graphs.")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()

# ================= HEADER =================
st.markdown("""
<div class="hero">
    <h1>🇮🇳 Indian Fake News Detection</h1>
    <p>Artificial Intelligence Powered News Verification Dashboard</p>
    <p>Logistic Regression + TF-IDF | Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ================= TOP METRICS =================
total = len(st.session_state.history)

real_count = sum(
    x["Prediction"] == "REAL"
    for x in st.session_state.history
)

fake_count = total - real_count

c1, c2, c3, c4 = st.columns(4)

c1.metric("🧪 Total Tests", total)
c2.metric("🟢 Real Predictions", real_count)
c3.metric("🔴 Fake Predictions", fake_count)
c4.metric("⚙️ Algorithm", "Logistic Regression")

st.markdown("---")

# ================= MAIN INPUT =================
left, right = st.columns([1.3, 1])

with left:
    st.markdown("## ✍️ Enter News Article")

    news = st.text_area(
        "Paste your Indian news article",
        height=250,
        placeholder="Enter complete English news article here..."
    )

    analyse = st.button(
        "🔍 ANALYSE NEWS",
        use_container_width=True
    )

    if analyse:

        if not news.strip():
            st.warning("Please enter news article.")

        else:
            cleaned = clean_text(news)

            if not cleaned:
                st.warning("Please enter valid English news text.")

            else:
                try:
                    features = vectorizer.transform([cleaned])

                    prediction = int(model.predict(features)[0])
                    probabilities = model.predict_proba(features)[0]

                    probs = {
                        int(label): float(prob)
                        for label, prob in zip(model.classes_, probabilities)
                    }

                    fake_prob = probs.get(0, 0) * 100
                    real_prob = probs.get(1, 0) * 100

                    label = "REAL" if prediction == 1 else "FAKE"
                    confidence = probs[prediction] * 100

                    st.session_state.result = {
                        "Prediction": label,
                        "Confidence": round(confidence, 2),
                        "Fake Probability": round(fake_prob, 2),
                        "Real Probability": round(real_prob, 2)
                    }

                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Prediction": label,
                        "Confidence": round(confidence, 2),
                        "Characters": len(news)
                    })

                except Exception as e:
                    st.error(f"Prediction error: {e}")

with right:
    st.markdown("## 📊 Prediction Result")

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="card">
            <h2>Waiting for News...</h2>
            <p>Enter an article and click Analyse News.</p>
            <h1>🧠 🔎 📰</h1>
        </div>
        """, unsafe_allow_html=True)

    else:

        if result["Prediction"] == "REAL":
            st.success("✅ PREDICTION: REAL NEWS")
        else:
            st.error("🚨 PREDICTION: FAKE NEWS")

        st.metric(
            "Model Confidence",
            str(result["Confidence"]) + "%"
        )

        st.markdown("### Probability Distribution")

        probability_df = pd.DataFrame({
            "Class": ["Fake", "Real"],
            "Probability": [
                result["Fake Probability"],
                result["Real Probability"]
            ]
        })

        st.bar_chart(
            probability_df.set_index("Class"),
            y="Probability",
            color="#38bdf8"
        )

        st.caption(
            "Model confidence is not proof of factual accuracy."
        )

# ================= ANALYTICS =================
st.markdown("---")
st.markdown("# 📈 MODEL TESTING ANALYTICS")

if st.session_state.history:

    df = pd.DataFrame(st.session_state.history)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Prediction Distribution")

        distribution = (
            df["Prediction"]
            .value_counts()
            .reindex(["REAL", "FAKE"], fill_value=0)
        )

        st.bar_chart(
            distribution,
            color="#22c55e"
        )

    with col2:
        st.markdown("### 📉 Confidence Analysis")

        confidence_df = df[["Confidence"]].copy()
        confidence_df.index = [
            f"Test {i+1}" for i in range(len(confidence_df))
        ]

        st.line_chart(
            confidence_df,
            color="#f59e0b"
        )

    st.markdown("### 📋 Prediction History")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Testing Report",
        data=csv,
        file_name="indian_fake_news_report.csv",
        mime="text/csv"
    )

else:
    st.info("Test news articles to generate analytics graphs.")

# ================= PROJECT INFORMATION =================
st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write("""
    Indian Fake News Detection is a machine learning
    project developed using Logistic Regression.

    Technologies:
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorizer
    - Streamlit

    Classification:
    - 0 = Fake News
    - 1 = Real News

    The application predicts based on learned text patterns.
    It does not independently verify news sources or facts.
    """)

st.markdown("""
<div style="text-align:center;color:#94a3b8;padding:25px;">
    🇮🇳 Indian Fake News Detection Project<br>
    Logistic Regression | TF-IDF | Binary Classification
</div>
""", unsafe_allow_html=True)

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Prediction distribution
st.subheader("🥧 Fake vs Real News Distribution")

if history:
    counts = pd.Series(
        [item["Prediction"] for item in history]
    ).value_counts()

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.45,
        color=counts.index,
        color_discrete_map={
            "REAL": "#00CC96",
            "FAKE": "#EF553B"
        }
    )

    fig.update_traces(
        textinfo="label+percent",
        pull=[0.08] * len(counts),
        rotation=45
    )

    fig.update_layout(
        title="News Prediction Overview",
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chart dekhne ke liye pehle news analyze karein.")

st.subheader("🎯 Model Confidence")

confidence = float(probability) * 100

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence,
    number={"suffix": "%"},
    title={"text": "Prediction Confidence"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#00CC96"},
        "steps": [
            {"range": [0, 50], "color": "#EF553B"},
            {"range": [50, 75], "color": "#FFA15A"},
            {"range": [75, 100], "color": "#00CC96"}
        ]
    }
))

fig.update_layout(
    template="plotly_dark",
    height=350
)

st.plotly_chart(fig, use_container_width=True)

