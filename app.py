import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Emotion detection
def detect_emotion(text):
    text = text.lower()
    if any(w in text for w in ["happy","great","love","excited","good"]):
        return "😊 Positive"
    elif any(w in text for w in ["sad","hopeless","tired","alone","worthless"]):
        return "😢 Sad"
    elif any(w in text for w in ["stress","anxious","worried","overthinking"]):
        return "😰 Anxiety"
    else:
        return "😐 Neutral"

# Keywords
def extract_keywords(text):
    keywords = ["hopeless","tired","alone","worthless","stress","anxious","sad"]
    return [w for w in keywords if w in text.lower()]

# Prediction
def predict(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    return pred, prob

# Page Config
st.set_page_config(page_title="AI Mental Health Analyzer", layout="wide")

# Custom Styling
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #22c55e;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background: #1e293b;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 AI Mental Health Analyzer")
st.caption("Smart text analysis using Machine Learning")

# Input Mode
mode = st.radio("Choose Mode", ["Single Text", "Multiple Texts"])

# -------- SINGLE --------
if mode == "Single Text":
    text = st.text_area("Enter your text")

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Enter some text")
        else:
            pred, prob = predict(text)
            emotion = detect_emotion(text)
            keywords = extract_keywords(text)

            st.markdown("### 🔍 Result")

            if pred == 1:
                st.markdown('<div class="result-box">⚠️ <b style="color:red;">HIGH RISK</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box">✅ <b style="color:lightgreen;">LOW RISK</b></div>', unsafe_allow_html=True)

            st.write(f"**Confidence:** {prob:.2f}")
            st.progress(float(prob))

            st.write(f"**Emotion:** {emotion}")

            if keywords:
                st.write(f"**Keywords detected:** {', '.join(keywords)}")
            else:
                st.write("**Keywords detected:** None")

# -------- MULTIPLE --------
else:
    multi_text = st.text_area("Enter multiple lines (one per line)")

    if st.button("Analyze All"):
        lines = multi_text.split("\n")

        for i, line in enumerate(lines):
            if line.strip():
                pred, prob = predict(line)
                emotion = detect_emotion(line)

                st.markdown(f"### Text {i+1}")
                st.write(line)

                if pred == 1:
                    st.error("HIGH RISK")
                else:
                    st.success("LOW RISK")

                st.write(f"Confidence: {prob:.2f}")
                st.write(f"Emotion: {emotion}")
                st.markdown("---")

# Footer
st.warning("⚠️ This is not a medical diagnosis tool.")