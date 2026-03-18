import pandas as pd
import torch
import joblib
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODELS_DIR, TEXT_COL, MAX_LENGTH

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


class Predictor:
    def __init__(self):
        self.sentiment_model_path = MODELS_DIR / "sentiment_model"
        self.aspect_model_path = MODELS_DIR / "aspect_model"
        self.aspect_encoder_path = MODELS_DIR / "aspect_label_encoder.pkl"

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_path)

        self.aspect_tokenizer = AutoTokenizer.from_pretrained(self.aspect_model_path)
        self.aspect_model = AutoModelForSequenceClassification.from_pretrained(self.aspect_model_path)

        self.aspect_encoder = joblib.load(self.aspect_encoder_path)

        self.sentiment_model.eval()
        self.aspect_model.eval()

    def predict_single(self, text: str):
        text = str(text).strip()

        sentiment_inputs = self.sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        aspect_inputs = self.aspect_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        with torch.no_grad():
            sentiment_outputs = self.sentiment_model(**sentiment_inputs)
            aspect_outputs = self.aspect_model(**aspect_inputs)

        sentiment_probs = softmax(sentiment_outputs.logits, dim=1)
        aspect_probs = softmax(aspect_outputs.logits, dim=1)

        sentiment_pred = torch.argmax(sentiment_probs, dim=1).item()
        aspect_pred = torch.argmax(aspect_probs, dim=1).item()

        sentiment_confidence = float(sentiment_probs[0][sentiment_pred].item())
        aspect_confidence = float(aspect_probs[0][aspect_pred].item())

        aspect_label = self.aspect_encoder.inverse_transform([aspect_pred])[0]

        return {
            "Review": text,
            "Sentiment_Code": sentiment_pred,
            "Sentiment_Label": SENTIMENT_MAP[sentiment_pred],
            "Sentiment_Confidence": round(sentiment_confidence, 4),
            "Aspect_Prediction": aspect_label,
            "Aspect_Confidence": round(aspect_confidence, 4)
        }

    def predict_dataframe(self, df: pd.DataFrame):
        results = []
        for text in df[TEXT_COL].astype(str):
            results.append(self.predict_single(text))
        return pd.DataFrame(results)