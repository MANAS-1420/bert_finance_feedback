import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TEXT_COL,
    ASPECT_COL,
    MODEL_NAME,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from src.utils import build_dataset, compute_metrics


def train_aspect():
    train_path = DATA_PROCESSED_DIR / "train.csv"
    test_path = DATA_PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test files not found. Run preprocess first.")

    train_df = pd.read_csv(train_path)[[TEXT_COL, ASPECT_COL]].dropna()
    test_df = pd.read_csv(test_path)[[TEXT_COL, ASPECT_COL]].dropna()

    label_encoder = LabelEncoder()
    train_df[ASPECT_COL] = train_df[ASPECT_COL].astype(str)
    test_df[ASPECT_COL] = test_df[ASPECT_COL].astype(str)

    train_df[ASPECT_COL] = label_encoder.fit_transform(train_df[ASPECT_COL])

    unseen_test_labels = set(test_df[ASPECT_COL]) - set(label_encoder.classes_)
    if unseen_test_labels:
        test_df = test_df[~test_df[ASPECT_COL].isin(unseen_test_labels)].copy()

    test_df[ASPECT_COL] = label_encoder.transform(test_df[ASPECT_COL])

    train_dataset = build_dataset(train_df, TEXT_COL, ASPECT_COL)
    test_dataset = build_dataset(test_df, TEXT_COL, ASPECT_COL)

    num_labels = len(label_encoder.classes_)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    output_dir = MODELS_DIR / "aspect_model"
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

    encoder_path = MODELS_DIR / "aspect_label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)

    print(f"Aspect model saved at: {output_dir}")
    print(f"Aspect label encoder saved at: {encoder_path}")


if __name__ == "__main__":
    train_aspect()