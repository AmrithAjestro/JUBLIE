# ----------------------------- train_model.py -----------------------------
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)
from dataset_loader import load_dataset

# ---------------- Paths ----------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "legal_classifier.pkl"
METRICS_LOG = MODEL_DIR / "metrics_log.csv"

def train_model():
    # ---------------- Load Dataset ----------------
    fj_df = load_dataset(
        "fj.csv",
        drop_columns=["judges name(s)", "cited cases", "Unnamed: 0"]
    )
    fj_df = fj_df.dropna(subset=["case title", "issues", "decision"])

    # Combine case title + issues into one text feature
    fj_df["text"] = fj_df["case title"] + " " + fj_df["issues"]
    X = fj_df["text"]
    y = fj_df["decision"]

    # ---------------- Train-Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- Pipeline ----------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # ---------------- Train ----------------
    print("🔹 Training model...")
    pipeline.fit(X_train, y_train)

    # ---------------- Predictions ----------------
    y_pred = pipeline.predict(X_test)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cls_report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"🎯 Macro F1-score: {f1_macro:.4f}")
    print(f"🎯 Weighted F1-score: {f1_weighted:.4f}")
    print("\n📊 Classification Report:\n", cls_report)

    # Confusion Matrix visualization
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)
    print("\n🧩 Confusion Matrix:\n", cm_df)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # ---------------- Save Model ----------------
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved at {MODEL_PATH}")

    # ---------------- Log Metrics ----------------
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

    if METRICS_LOG.exists():
        metrics_df = pd.read_csv(METRICS_LOG)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([log_data])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([log_data])

    metrics_df.to_csv(METRICS_LOG, index=False)
    print(f"📈 Metrics logged to {METRICS_LOG}")

if __name__ == "__main__":
    train_model()
