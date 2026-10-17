"""
Movie Genre Prediction using Wikipedia Movie Plots dataset.

Expected CSV:
data/dataset.csv

Dataset columns used:
Plot  -> input text
Genre -> target class
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Set your downloaded dataset path here.
DATA_PATH = Path("data/dataset.csv")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load only the two columns needed.
df = pd.read_csv(DATA_PATH, usecols=["Genre", "Plot"])
df = df.dropna(subset=["Genre", "Plot"])
df["Genre"] = df["Genre"].astype(str).str.strip()
df["Plot"] = df["Plot"].astype(str).str.strip()
df = df[(df["Genre"] != "") & (df["Plot"] != "") & (df["Genre"].str.lower() != "unknown")]
df = df.drop_duplicates(subset="Plot")

# Keep genres with at least 20 examples for a more reliable train/test evaluation.
counts = df["Genre"].value_counts()
df = df[df["Genre"].isin(counts[counts >= 20].index)]

X_train, X_test, y_train, y_test = train_test_split(
    df["Plot"], df["Genre"], test_size=0.20, random_state=42, stratify=df["Genre"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        max_features=100_000, min_df=2, sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1500, class_weight="balanced"
    ))
])

print(f"Movies used: {len(df):,}")
print(f"Genres used: {df['Genre'].nunique()}")
print("Training model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
    "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
}
print("\nEvaluation:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

report = classification_report(y_test, predictions, zero_division=0)
print("\nClassification report:\n", report)

joblib.dump(model, OUTPUT_DIR / "movie_genre_model.joblib")
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# Plot the most common genres in the filtered dataset.
df["Genre"].value_counts().head(20).sort_values().plot(kind="barh", figsize=(9, 6))
plt.title("Top 20 Genres in Wikipedia Movie Plots")
plt.xlabel("Number of movies")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_distribution.png", dpi=150)
plt.close()

# Test a new plot description.
sample_plot = (
    "A detective investigates a series of mysterious murders and uncovers "
    "a dangerous conspiracy in a small town."
)
print("\nSample plot:", sample_plot)
print("Predicted genre:", model.predict([sample_plot])[0])
print(f"\nSaved model and reports to: {OUTPUT_DIR.resolve()}")