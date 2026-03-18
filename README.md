# BERT/XLM-R Finance Customer Feedback Intelligence

## Overview
This project is an end-to-end NLP application for finance customer reviews. It predicts:

- Sentiment: Negative / Neutral / Positive
- Aspect/Category: staff, service, charges, delay, digital_app, documentation, etc.

The model is built using `xlm-roberta-base`, which supports multilingual text and works well for English, Hindi, and Hinglish reviews.

## Features
- Sentiment classification (0, 1, 2)
- Aspect classification
- Confidence score
- CSV preprocessing
- Streamlit web app
- Single prediction and batch prediction
- GitHub-ready project structure

## Tech Stack
- Python
- pandas
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Streamlit

## Dataset Format

Your CSV should contain:

- `Review`
- `Sentiment`
- `Aspect`

Example:

```csv
Review,Sentiment,Aspect
"The staff was very helpful",2,staff
"Loan processing took too long",0,delay
"Charges were okay",1,charges