import streamlit as st
import pandas as pd
from src.predict import Predictor

st.set_page_config(page_title="Finance Feedback Intelligence", layout="wide")
st.title("Finance Customer Feedback Intelligence System")
st.write("Predict sentiment and aspect from finance customer reviews.")

@st.cache_resource
def load_predictor():
    return Predictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error("Models are not ready yet. Please run preprocessing and training first.")
    st.exception(e)
    st.stop()

tab1, tab2 = st.tabs(["Single Review Prediction", "Batch CSV Prediction"])

with tab1:
    st.subheader("Single Review Prediction")
    user_text = st.text_area("Enter review text")

    if st.button("Analyze Review"):
        if user_text.strip():
            result = predictor.predict_single(user_text)

            st.success("Prediction completed")
            st.write("### Result")
            st.write(f"**Sentiment Code:** {result['Sentiment_Code']}")
            st.write(f"**Sentiment Label:** {result['Sentiment_Label']}")
            st.write(f"**Sentiment Confidence:** {result['Sentiment_Confidence']}")
            st.write(f"**Aspect:** {result['Aspect_Prediction']}")
            st.write(f"**Aspect Confidence:** {result['Aspect_Confidence']}")
        else:
            st.warning("Please enter a review.")

with tab2:
    st.subheader("Batch CSV Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "Review" not in df.columns:
            st.error("CSV must contain a 'Review' column.")
        else:
            predictions = predictor.predict_dataframe(df)
            final_df = pd.concat(
                [df.reset_index(drop=True), predictions.drop(columns=["Review"])],
                axis=1
            )

            st.write("### Prediction Output")
            st.dataframe(final_df, use_container_width=True)

            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_data,
                file_name="predicted_reviews.csv",
                mime="text/csv"
            )