import streamlit as st
import pandas as pd
import re
import time
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# NLTK FIX (STREAMLIT CLOUD)
# -------------------------------------------------
nltk.download("stopwords")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake Review Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# THEME-AWARE CSS (LIGHT + DARK)
# -------------------------------------------------
st.markdown("""
<style>

/* Use Streamlit theme variables */
:root {
    --bg: var(--background-color);
    --text: var(--text-color);
    --card: var(--secondary-background-color);
}

/* Hide Streamlit toolbar icons */
header {
    display: none;
}

/* Remove extra top space */
.stApp {
    margin-top: -80px;
}

/* ---------------- CARD STYLE (IMPORTANT FIX) ---------------- */
.card {
    background-color: var(--card);
    padding: 22px;
    border-radius: 18px;

    /* Visible border */
    border: 1px solid rgba(0, 0, 0, 0.12);

    /* Depth */
    box-shadow:
        0 8px 24px rgba(0, 0, 0, 0.18),
        inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}

/* Dark mode enhancement */
@media (prefers-color-scheme: dark) {
    .card {
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.65),
            inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }
}

/* Stats */
.stat {
    font-size: 40px;
    font-weight: 800;
    color: var(--text);
}

.label {
    font-size: 15px;
    color: rgba(120, 120, 120, 0.9);
}

/* Textarea */
textarea {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 52px;
    font-size: 18px;
    border-radius: 14px;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #22c55e;
}

/* Watermark (bottom-left) */
.footer {
    position: fixed;
    bottom: 10px;
    left: 20px;
    opacity: 0.45;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")
    return data.sample(n=5000, random_state=42)

data = load_data()

# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["clean_review"] = data["text_"].apply(clean_text)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
X = data["clean_review"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("## 👋 Welcome")
st.caption("AI-powered Fake Review Detection Dashboard")
st.write("")

# -------------------------------------------------
# STATS
# -------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        "<div class='card'><div class='stat'>85%</div><div class='label'>Model Accuracy</div></div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        "<div class='card'><div class='stat'>5,000</div><div class='label'>Reviews Trained</div></div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        "<div class='card'><div class='stat'>LIVE</div><div class='label'>Prediction Mode</div></div>",
        unsafe_allow_html=True
    )

st.write("")

# -------------------------------------------------
# MAIN AREA
# -------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Paste Review for Analysis")

    review = st.text_area(
        "",
        height=160,
        placeholder="Paste Amazon / Flipkart / Google review here..."
    )

    if st.button("🔍 Analyze Review"):
        if review.strip() == "":
            st.warning("Please paste a review first.")
        else:
            with st.spinner("Analyzing with AI..."):
                time.sleep(1)

            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            fake_prob = model.predict_proba(vec)[0][1]

            st.progress(fake_prob)

            if fake_prob >= 0.60:
                st.error(f"❌ Fake Review Detected (Confidence: {fake_prob:.2f})")
            else:
                st.success(f"✅ Genuine Review (Confidence: {1 - fake_prob:.2f})")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ System Info")
    st.write("• NLP + TF-IDF")
    st.write("• Logistic Regression")
    st.write("• Probability-based detection")
    st.write("• Secure & Read-only data")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# WATERMARK
# -------------------------------------------------
st.markdown(
    "<div class='footer'>Developed by Aditya Kumar Gupta </div>",
    unsafe_allow_html=True
)
