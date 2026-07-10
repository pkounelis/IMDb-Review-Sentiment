"""Train sklearn models with Pipelines, GridSearchCV, and a comparison loop."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import (
    COUNT_MAX_FEATURES,
    GRID_CV_FOLDS,
    LOGREG_C_GRID,
    MODELS_DIR,
    RF_DEPTH_GRID,
    RF_ESTIMATORS_GRID,
    SEED,
    TFIDF_MAX_FEATURES,
    TRAIN_FIT_CSV,
    VAL_CSV,
)


def load_xy(csv_path, text_col: str = "text_sklearn"):
    df = pd.read_csv(csv_path)
    return df[text_col], df["label"]


def evaluate_pipeline(name: str, pipeline, x_val, y_val) -> float:
    preds = pipeline.predict(x_val)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"{name:32s}  val macro-F1 = {f1:.4f}")
    return f1


def main() -> None:
    if not TRAIN_FIT_CSV.exists():
        raise FileNotFoundError(
            f"{TRAIN_FIT_CSV} not found. Run: python -m src.preprocess"
        )

    x_train, y_train = load_xy(TRAIN_FIT_CSV)
    x_val, y_val = load_xy(VAL_CSV)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipelines: dict[str, Pipeline | GridSearchCV] = {
        "bow_logreg": Pipeline(
            [
                ("vectorizer", CountVectorizer(max_features=COUNT_MAX_FEATURES)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=SEED),
                ),
            ]
        ),
        "tfidf_logreg": GridSearchCV(
            Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=TFIDF_MAX_FEATURES,
                            ngram_range=(1, 2),
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1000, random_state=SEED),
                    ),
                ]
            ),
            param_grid={"classifier__C": LOGREG_C_GRID},
            cv=GRID_CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1,
        ),
        "bow_rf": Pipeline(
            [
                ("vectorizer", CountVectorizer(max_features=COUNT_MAX_FEATURES)),
                (
                    "classifier",
                    RandomForestClassifier(random_state=SEED, n_jobs=-1),
                ),
            ]
        ),
        "tfidf_rf": GridSearchCV(
            Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            max_features=TFIDF_MAX_FEATURES,
                            ngram_range=(1, 2),
                        ),
                    ),
                    (
                        "classifier",
                        RandomForestClassifier(random_state=SEED, n_jobs=-1),
                    ),
                ]
            ),
            param_grid={
                "classifier__max_depth": RF_DEPTH_GRID,
                "classifier__n_estimators": RF_ESTIMATORS_GRID,
            },
            cv=GRID_CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1,
        ),
    }

    display_names = {
        "bow_logreg": "BoW + LogReg (baseline)",
        "bow_rf": "BoW + RandomForest (baseline)",
        "tfidf_logreg": "TF-IDF + LogReg (tuned)",
        "tfidf_rf": "TF-IDF + RandomForest (tuned)",
    }

    results: dict[str, float] = {}
    best_params: dict[str, dict] = {}

    for key, estimator in pipelines.items():
        name = display_names[key]
        print(f"\n--- Training {name} ---")
        estimator.fit(x_train, y_train)
        fitted = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
        results[key] = evaluate_pipeline(name, fitted, x_val, y_val)
        joblib.dump(fitted, MODELS_DIR / f"{key}.pkl")
        if hasattr(estimator, "best_params_"):
            best_params[key] = estimator.best_params_
            print(f"  best params: {estimator.best_params_}")

    with open(MODELS_DIR / "sklearn_val_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(MODELS_DIR / "sklearn_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    best = max(results, key=results.get)
    print(f"\nBest sklearn model on val: {display_names[best]} (F1={results[best]:.4f})")


if __name__ == "__main__":
    main()
