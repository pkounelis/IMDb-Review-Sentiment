"""Load IMDb from HuggingFace, clean text, and save train/val/test CSV splits."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED,
    SEED,
    TEST_CSV,
    TRAIN_FIT_CSV,
    VAL_CSV,
    VAL_SIZE,
)
from src.text_clean import clean_for_sentiment, clean_for_sklearn


def load_imdb_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return HF train and test splits as pandas DataFrames."""
    imdb = load_dataset("stanfordnlp/imdb")
    train_df = imdb["train"].to_pandas()
    test_df = imdb["test"].to_pandas()
    return train_df, test_df


def build_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split HF train into train_fit + val; keep HF test untouched."""
    train_df, test_df = load_imdb_splits()

    train_fit, val = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        stratify=train_df["label"],
        random_state=SEED,
    )

    for frame in (train_fit, val, test_df):
        frame["text_sklearn"] = frame["text"].map(clean_for_sklearn)
        frame["text_sentiment"] = frame["text"].map(clean_for_sentiment)

    return (
        train_fit.reset_index(drop=True),
        val.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_fit: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_fit.to_csv(TRAIN_FIT_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)


def main() -> None:
    train_fit, val, test = build_splits()
    save_splits(train_fit, val, test)
    print(f"Saved train_fit: {len(train_fit):,} rows -> {TRAIN_FIT_CSV}")
    print(f"Saved val:       {len(val):,} rows -> {VAL_CSV}")
    print(f"Saved test:      {len(test):,} rows -> {TEST_CSV}")
    print("\nLabel distribution (train_fit):")
    print(train_fit["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
