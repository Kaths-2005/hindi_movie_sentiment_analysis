"""
model.py
--------
Trains (or loads a cached) TF-IDF + Logistic Regression sentiment model.

The dataset has no ground-truth sentiment labels, so weak-supervision
labels are generated from an emoji + Hindi keyword lexicon and used as
the training target. This is disclosed to the user in the UI/README --
it is NOT a claim of human-annotated ground truth.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from preprocessing import load_and_clean_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
DATASET_CACHE_PATH = os.path.join(MODELS_DIR, "dataset_with_sentiment.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.pkl")

POSITIVE_EMOJIS = {"❤", "😍", "😊", "😄", "😁", "🔥", "👍", "💪", "🙏", "🎉",
                    "✨", "🥰", "😎", "👏", "✅", "✊", "🎶", "🤣", "😂", "🥳"}
NEGATIVE_EMOJIS = {"😢", "😡", "😤", "🙁", "👎", "🖕", "🤬", "🤮", "🤦",
                    "🥵", "🥶", "💋", "👄", "👙", "🍌", "🤤", "😮"}

POSITIVE_WORDS = [
    "अच्छा", "अच्छी", "अच्छे", "बेहतरीन", "शानदार", "जबरदस्त", "सुपरहिट", "हिट",
    "खुबसूरत", "खूबसूरत", "प्यार", "धन्यवाद", "बधाई", "जय", "शुभकामनाएं", "मस्त",
    "लाजवाब", "ज़बरदस्त", "कमाल", "सुंदर", "बेस्ट", "पसंद", "एन्जॉय", "ऐन्जॉय",
    "दमदार", "सही", "वाह", "जियो", "सपोर्ट", "आग", "हैप्पी", "खुश",
]
NEGATIVE_WORDS = [
    "बकवास", "बेकार", "गंदा", "गंदी", "फ्लॉप", "घटिया", "शर्म", "शर्मनाक", "नफरत",
    "गलत", "बुरा", "बुरी", "दर्द", "जुल्म", "सर्वनाश", "मौत", "रिजेक्ट", "गिराने",
    "बदला", "खून", "बदतमीज", "नाराज", "दुख", "तड़पाया", "क्रूर", "जेहाद",
]


def _emoji_score(s):
    if not isinstance(s, str):
        return 0
    return sum(1 if ch in POSITIVE_EMOJIS else (-1 if ch in NEGATIVE_EMOJIS else 0) for ch in s)


def _text_score(t):
    if not isinstance(t, str):
        return 0
    score = sum(1 for w in POSITIVE_WORDS if w in t)
    score -= sum(1 for w in NEGATIVE_WORDS if w in t)
    return score


def weak_label(row):
    score = _text_score(row["final_preprocessed_text"]) + _emoji_score(row.get("emojis_extracted", ""))
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def train(dataset_path: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_clean_dataset(dataset_path)
    df["weak_label"] = df.apply(weak_label, axis=1)

    X_text = df["final_preprocessed_text"]
    y = df["weak_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=800, min_df=1, token_pattern=r"[^\s]+")
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    labels_order = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "labels": labels_order,
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    # Predict sentiment for the whole dataset with the trained model
    X_all = vectorizer.transform(X_text)
    df["sentiment"] = clf.predict(X_all)
    proba = clf.predict_proba(X_all)
    df["sentiment_confidence"] = [round(float(max(p)), 4) for p in proba]

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)
    df.to_csv(DATASET_CACHE_PATH, index=False, encoding="utf-8-sig")

    return vectorizer, clf, df, metrics


def load_or_train(dataset_path: str, force_retrain: bool = False):
    """Load cached model/vectorizer/dataset if present, else train fresh."""
    if (
        not force_retrain
        and os.path.exists(VECTORIZER_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(DATASET_CACHE_PATH)
        and os.path.exists(METRICS_PATH)
    ):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
        df = pd.read_csv(DATASET_CACHE_PATH)
        return vectorizer, clf, df, metrics

    return train(dataset_path)


def predict_review(vectorizer, clf, text: str):
    from preprocessing import preprocess_for_model
    cleaned = preprocess_for_model(text)
    if not cleaned:
        cleaned = text
    X = vectorizer.transform([cleaned])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    confidence = float(max(proba))

    # lexicon fallback / tie-break, same idea as the browser-side model
    lex_score = _text_score(cleaned) + _emoji_score(text)
    if X.nnz == 0 and lex_score != 0:
        pred = "positive" if lex_score > 0 else "negative"
        confidence = min(0.6 + abs(lex_score) * 0.08, 0.95)
    elif lex_score > 0 and pred == "negative":
        pred = "positive"
        confidence = max(confidence, 0.55)
    elif lex_score < 0 and pred == "positive":
        pred = "negative"
        confidence = max(confidence, 0.55)

    return pred, confidence
