"""Evaluate all models on the held-out HF test split and save plots."""

from __future__ import annotations

import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    HF_DISPLAY_NAME,
    LABEL_NAMES,
    MODELS_DIR,
    PLOTS_DIR,
    SENTIMENT_MODEL_NAME,
    TEST_CSV,
)
from src.hf_sentiment import predict_sentiment

SKLEARN_MODELS = {
    "BoW + LogReg": MODELS_DIR / "bow_logreg.pkl",
    "BoW + RandomForest": MODELS_DIR / "bow_rf.pkl",
    "TF-IDF + LogReg": MODELS_DIR / "tfidf_logreg.pkl",
    "TF-IDF + RandomForest": MODELS_DIR / "tfidf_rf.pkl",
}


def slugify(name: str) -> str:
    slug = name.lower().replace("+", "plus").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", slug)


def predict_sklearn(pipeline, texts: pd.Series) -> np.ndarray:
    return pipeline.predict(texts)


def predict_sklearn_proba(pipeline, texts: pd.Series) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.predict(texts)


def save_confusion_matrix(y_true, y_pred, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_combined_roc(y_true, model_probs: dict[str, np.ndarray], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in model_probs.items():
        if len(np.unique(probs)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — all models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"{TEST_CSV} not found. Run: python -m src.preprocess")

    test_df = pd.read_csv(TEST_CSV)
    y_test = test_df["label"]
    texts_sklearn = test_df["text_sklearn"]
    texts_sentiment = test_df["text_sentiment"].tolist()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    all_probs: dict[str, np.ndarray] = {}
    best_f1 = -1.0
    best_name = ""

    for name, path in SKLEARN_MODELS.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found (run train_sklearn.py)")
            continue
        pipeline = joblib.load(path)
        preds = predict_sklearn(pipeline, texts_sklearn)
        probs = predict_sklearn_proba(pipeline, texts_sklearn)
        f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(np.unique(probs)) > 1 else None
        results[name] = {"f1": f1, "accuracy": acc, "roc_auc": auc}
        all_probs[name] = probs

        cm_path = PLOTS_DIR / f"confusion_matrix_{slugify(name)}.png"
        save_confusion_matrix(y_test, preds, f"Confusion Matrix — {name}", cm_path)

        print(f"\n=== {name} (test) ===")
        print(classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    print(f"\nLoading pretrained model: {SENTIMENT_MODEL_NAME}")
    preds, probs = predict_sentiment(texts_sentiment)
    f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results[HF_DISPLAY_NAME] = {"f1": f1, "accuracy": acc, "roc_auc": auc}
    all_probs[HF_DISPLAY_NAME] = probs

    cm_path = PLOTS_DIR / f"confusion_matrix_{slugify(HF_DISPLAY_NAME)}.png"
    save_confusion_matrix(
        y_test, preds, f"Confusion Matrix — {HF_DISPLAY_NAME}", cm_path
    )

    print(f"\n=== {HF_DISPLAY_NAME} (test) ===")
    print(classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0))
    if f1 > best_f1:
        best_f1 = f1
        best_name = HF_DISPLAY_NAME

    if all_probs:
        roc_path = PLOTS_DIR / "roc_combined.png"
        save_combined_roc(y_test, all_probs, roc_path)
        print(f"\nCombined ROC saved to {roc_path}")

    if best_name:
        print(f"\nBest model on test: {best_name} (F1={best_f1:.4f})")
        print(f"Per-model confusion matrices saved to {PLOTS_DIR}")

    with open(MODELS_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print("\nResults summary:")
    for name, metrics in results.items():
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "—"
        print(
            f"  {name:24s}  F1={metrics['f1']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  AUC={auc_str}"
        )


if __name__ == "__main__":
    main()
