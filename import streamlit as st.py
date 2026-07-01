import streamlit as st
from model import predict

st.set_page_config(page_title="AI Mental Health Analyzer PRO", page_icon="🧠", layout="centered")

st.title("🧠 AI Mental Health Analyzer PRO")
st.markdown("Advanced AI system for mental health risk & emotion detection")

user_input = st.text_area("Enter your thoughts:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        label, confidence = predict(user_input)

        st.subheader("Result:")

        if label == "High Risk":
            st.error(f"🔴 {label}")
        elif label == "Moderate Risk":
            st.warning(f"🟠 {label}")
        else:
            st.success(f"🟢 {label}")

        st.write("Confidence Score:", round(confidence * 100, 2), "%")

st.markdown("---")
st.caption("⚠ Educational project only – not a medical diagnosis")

