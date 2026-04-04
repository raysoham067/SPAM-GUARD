"""
Run this ONCE on your local machine to generate the sklearn models:
    python train_models.py

This creates:
    models/dense_model.pkl
    models/use_model.pkl
"""

import os, joblib, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import fetch_20newsgroups   # only used as fallback
import urllib.request, csv, io

os.makedirs("models", exist_ok=True)

# ── Download UCI SMS Spam dataset ──────────────────────────────────────────
print("Downloading SMS Spam dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
try:
    with urllib.request.urlopen(url) as r:
        content = r.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(content), delimiter="\t"))
    labels = [1 if r[0].strip() == "spam" else 0 for r in rows]
    texts  = [r[1] for r in rows]
    print(f"  Loaded {len(texts)} messages ({sum(labels)} spam)")
except Exception as e:
    print(f"  Download failed ({e}), using built-in fallback data...")
    # Minimal fallback so script doesn't crash
    texts = [
        "WINNER!! You've won a free prize call now",
        "Congratulations! Claim your FREE gift today",
        "URGENT: Your account has been suspended click here",
        "Win £1000 cash call 09061743810 from landline",
        "Free entry in 2 a weekly comp to win FA Cup",
        "Hey are you coming to the meeting today?",
        "Can you pick up some milk on the way home?",
        "I'll be late for dinner, stuck in traffic",
        "Happy birthday! Hope you have a great day",
        "Let's catch up this weekend, it's been a while",
        "Your appointment is confirmed for tomorrow at 3pm",
        "Did you finish the report? The boss is asking",
    ] * 50
    labels = ([1]*5 + [0]*7) * 50

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── Model 1: Dense Network equivalent (TF-IDF + Logistic Regression) ──────
print("Training Dense Network equivalent (TF-IDF + LogisticRegression)...")
dense_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                               sublinear_tf=True, strip_accents="unicode")),
    ("clf",   LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs"))
])
dense_pipe.fit(X_train, y_train)
acc = dense_pipe.score(X_test, y_test)
print(f"  Dense model accuracy: {acc*100:.1f}%")
joblib.dump(dense_pipe, "models/dense_model.pkl")
print("  Saved → models/dense_model.pkl")

# ── Model 2: USE equivalent (TF-IDF char + GradientBoosting) ──────────────
print("Training USE equivalent (TF-IDF char-ngram + GradientBoosting)...")
use_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=8000, analyzer="char_wb",
                               ngram_range=(3,5), sublinear_tf=True)),
    ("clf",   LogisticRegression(C=3.0, max_iter=1000, solver="lbfgs"))
])
use_pipe.fit(X_train, y_train)
acc = use_pipe.score(X_test, y_test)
print(f"  USE model accuracy: {acc*100:.1f}%")
joblib.dump(use_pipe, "models/use_model.pkl")
print("  Saved → models/use_model.pkl")

print("\n✅ Done! Both models saved in models/ folder.")
print("Now push everything to GitHub and redeploy on Streamlit Cloud.")