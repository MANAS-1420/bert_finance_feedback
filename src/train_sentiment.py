import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TEXT_COL,
    SENTIMENT_COL,
    MODEL_NAME,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from src.utils import build_dataset, compute_metrics


def train_sentiment():
    train_path = DATA_PROCESSED_DIR / "train.csv"
    test_path = DATA_PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test files not found. Run preprocess first.")

    train_df = pd.read_csv(train_path)[[TEXT_COL, SENTIMENT_COL]].dropna()
    test_df = pd.read_csv(test_path)[[TEXT_COL, SENTIMENT_COL]].dropna()

    train_dataset = build_dataset(train_df, TEXT_COL, SENTIMENT_COL)
    test_dataset = build_dataset(test_df, TEXT_COL, SENTIMENT_COL)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    output_dir = MODELS_DIR / "sentiment_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    print(f"Sentiment model saved at: {output_dir}")


if __name__ == "__main__":
    train_sentiment()