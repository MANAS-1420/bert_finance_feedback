import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def build_dataset(df: pd.DataFrame, text_col: str, label_col: str):
    dataset = Dataset.from_pandas(df[[text_col, label_col]].copy())

    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column(label_col, "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}