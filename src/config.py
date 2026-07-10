"""Shared paths and hyperparameters for the IMDb sentiment pipeline."""

from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"

TRAIN_FIT_CSV = DATA_PROCESSED / "train_fit.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

LABEL_NAMES = ["negative", "positive"]
VAL_SIZE = 0.15

# Sklearn vectorizers
COUNT_MAX_FEATURES = 10_000
TFIDF_MAX_FEATURES = 15_000

# GridSearchCV (Medium tier)
GRID_CV_FOLDS = 3
LOGREG_C_GRID = [0.1, 1.0, 10.0]
RF_DEPTH_GRID = [None, 30]
RF_ESTIMATORS_GRID = [100, 150]

# HuggingFace pretrained sentiment model (SST-2 binary)
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_DISPLAY_NAME = "HF SST-2 DistilBERT"
MAX_LENGTH = 512
HF_BATCH_SIZE = 32
