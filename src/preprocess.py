import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TEXT_COL,
    SENTIMENT_COL,
    ASPECT_COL,
    RANDOM_STATE,
    TEST_SIZE
)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_data(input_filename="finance_reviews.csv"):
    input_path = DATA_RAW_DIR / input_filename

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [TEXT_COL, SENTIMENT_COL, ASPECT_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=[TEXT_COL, SENTIMENT_COL, ASPECT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
    df = df.dropna(subset=[SENTIMENT_COL]).copy()
    df[SENTIMENT_COL] = df[SENTIMENT_COL].astype(int)
    df = df[df[SENTIMENT_COL].isin([0, 1, 2])].copy()
    df[ASPECT_COL] = df[ASPECT_COL].astype(str).str.strip().str.lower()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = DATA_PROCESSED_DIR / "cleaned_reviews.csv"
    df.to_csv(cleaned_path, index=False)

    # Remove rare aspect labels that appear only once
    aspect_counts = df[ASPECT_COL].value_counts()
    valid_aspects = aspect_counts[aspect_counts >= 2].index
    df = df[df[ASPECT_COL].isin(valid_aspects)].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[SENTIMENT_COL]
    )

    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    print("Preprocessing completed successfully.")
    print(f"Total rows after cleaning: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved cleaned data to: {cleaned_path}")


if __name__ == "__main__":
    preprocess_data()