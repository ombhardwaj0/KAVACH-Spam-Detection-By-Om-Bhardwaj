import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_dataset():
    """Load either the common Kaggle CSV or the UCI tab-separated file."""
    candidates = [
        os.path.join(DATASET_DIR, "spam.csv"),
        os.path.join(DATASET_DIR, "SMSSpamCollection"),
        os.path.join(DATASET_DIR, "SMSSpamCollection.txt"),
    ]

    file_path = next((p for p in candidates if os.path.isfile(p)), None)
    if file_path is None:
        raise FileNotFoundError(
            "Dataset not found. Put spam.csv or SMSSpamCollection "
            "inside the dataset folder. See README.md."
        )

    # Kaggle-style CSV: columns v1 (label), v2 (message)
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path, encoding="latin-1")
        if {"v1", "v2"}.issubset(df.columns):
            df = df[["v1", "v2"]].copy()
            df.columns = ["label", "message"]
        elif {"label", "message"}.issubset(df.columns):
            df = df[["label", "message"]].copy()
        elif {"label", "text"}.issubset(df.columns):
            df = df[["label", "text"]].copy()
            df.columns = ["label", "message"]
        else:
            raise ValueError(
                "CSV columns not recognized. Expected v1/v2, label/message, "
                "or label/text."
            )
    else:
        # UCI original file has no header and is tab-separated.
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1",
        )

    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset()
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates()

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both ham and spam messages.")

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", MultinomialNB()),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted; Ham, Spam):")
    print(matrix)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "spam_precision": float(report["Spam"]["precision"]),
        "spam_recall": float(report["Spam"]["recall"]),
        "spam_f1_score": float(report["Spam"]["f1-score"]),
        "total_messages": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "confusion_matrix": matrix.tolist(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
