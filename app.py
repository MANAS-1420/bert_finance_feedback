import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Finance Feedback Analyzer", layout="wide")

st.title("💰 Finance Customer Feedback Analyzer")
st.write("AI-based Sentiment Analysis using HuggingFace")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

user_text = st.text_area("Enter customer review")

if st.button("Analyze"):
    if user_text.strip():
        result = model(user_text)[0]

        st.success("Analysis Done ✅")
        st.write(f"**Label:** {result['label']}")
        st.write(f"**Confidence:** {round(result['score'], 4)}")
    else:
        st.warning("Please enter text")
