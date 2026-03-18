from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEXT_COL = "Review"
SENTIMENT_COL = "Sentiment"
ASPECT_COL = "Aspect"

MODEL_NAME = "xlm-roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
TEST_SIZE = 0.2