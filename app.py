import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Multilingual Finance Feedback Analyzer", layout="wide")

st.title("💰 Multilingual Finance Feedback Analyzer")
st.write("Sentiment analysis for English, Hindi, and Hinglish reviews")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

model = load_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

user_text = st.text_area("Enter customer review")

if st.button("Analyze"):
    if user_text.strip():
        result = model(user_text)[0]
        label = label_map.get(result["label"], result["label"])

        st.success("Analysis Done ✅")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {round(result['score'], 4)}")
    else:
        st.warning("Please enter text")
