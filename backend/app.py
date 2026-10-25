"""
Flask backend for Hindi Movie Sentiment Analysis.
Adds persistent user accounts and user-submitted reviews using SQLite.
"""
import os
import sqlite3
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from model import load_or_train, predict_review
from recommendation import build_movie_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "movie_reviews.csv")
DB_PATH = os.path.join(BASE_DIR, "movie_sentiment.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "hindi-movie-sentiment-dev-secret-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app, supports_credentials=True)


def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(movie_id) REFERENCES movies(id)
        );
    """)
    conn.commit()
    conn.close()


def seed_movie_names():
    if DATAFRAME is None:
        return
    conn = db_connection()
    now = datetime.utcnow().isoformat(timespec="seconds")
    for name in DATAFRAME["movie_name"].dropna().unique():
        conn.execute("INSERT OR IGNORE INTO movies(movie_name, created_at) VALUES (?, ?)", (str(name), now))
    conn.commit()
    conn.close()


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "इस सुविधा के लिए पहले लॉग इन करें।"}), 401
        return fn(*args, **kwargs)
    return wrapped


print("मॉडल लोड/प्रशिक्षित किया जा रहा है (loading/training model)...")
try:
    VECTORIZER, MODEL, DATAFRAME, METRICS = load_or_train(DATASET_PATH)
    LOAD_ERROR = None
    init_db()
    seed_movie_names()
    print(f"तैयार: {len(DATAFRAME)} समीक्षाएँ, {DATAFRAME['movie_name'].nunique()} फिल्में।")
except Exception as exc:
    VECTORIZER = MODEL = DATAFRAME = METRICS = None
    LOAD_ERROR = str(exc)
    print("डेटासेट लोड नहीं हो सका:", LOAD_ERROR)


def dataset_guard():
    if LOAD_ERROR or DATAFRAME is None:
        return jsonify({"error": "डेटासेट लोड नहीं हो सका। कृपया डेटासेट की जाँच करें।"}), 500
    return None


def user_reviews_dataframe():
    conn = db_connection()
    rows = conn.execute("""
        SELECT r.id, m.movie_name, r.review_text AS raw_text,
               r.sentiment, r.confidence AS sentiment_confidence,
               r.created_at, u.username
        FROM reviews r
        JOIN movies m ON m.id = r.movie_id
        JOIN users u ON u.id = r.user_id
        ORDER BY r.id DESC
    """).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["movie_name", "raw_text", "sentiment", "sentiment_confidence", "created_at", "username"])
    return pd.DataFrame([dict(r) for r in rows])


def combined_dataframe():
    base = DATAFRAME[["movie_name", "raw_text", "sentiment", "sentiment_confidence"]].copy()
    user_df = user_reviews_dataframe()
    if not user_df.empty:
        user_df = user_df[["movie_name", "raw_text", "sentiment", "sentiment_confidence"]]
        base = pd.concat([base, user_df], ignore_index=True)
    return base


def current_movie_stats():
    return build_movie_stats(combined_dataframe())


def find_movie(movie_name):
    target = movie_name.strip().casefold()
    return next((m for m in current_movie_stats() if m["movie_name"].casefold() == target), None)


@app.route("/api/auth/register", methods=["POST"])
def register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    confirm = payload.get("confirm_password", password)
    if not username or not password:
        return jsonify({"error": "कृपया उपयोगकर्ता नाम और पासवर्ड भरें।"}), 400
    if len(username) < 2:
        return jsonify({"error": "उपयोगकर्ता नाम कम से कम 2 अक्षरों का होना चाहिए।"}), 400
    if len(password) < 6:
        return jsonify({"error": "पासवर्ड कम से कम 6 अक्षरों का होना चाहिए।"}), 400
    if password != confirm:
        return jsonify({"error": "दोनों पासवर्ड समान होने चाहिए।"}), 400
    conn = db_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.utcnow().isoformat(timespec="seconds")),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "यह उपयोगकर्ता नाम पहले से मौजूद है।"}), 409
    conn.close()
    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"message": "खाता सफलतापूर्वक बनाया गया।", "username": username})


@app.route("/api/auth/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    conn = db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "उपयोगकर्ता नाम या पासवर्ड गलत है।"}), 401
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"message": "लॉग इन सफल रहा।", "username": user["username"]})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "आप सफलतापूर्वक लॉग आउट हो गए।"})


@app.route("/api/auth/me", methods=["GET"])
def me():
    if not session.get("user_id"):
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": session.get("username")})


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    err = dataset_guard()
    if err:
        return err
    combined = combined_dataframe()
    stats = build_movie_stats(combined)
    total = len(combined)
    pos = int((combined["sentiment"] == "positive").sum())
    neg = int((combined["sentiment"] == "negative").sum())
    neu = int((combined["sentiment"] == "neutral").sum())
    avg_watch = round(sum(m["watchability_score"] for m in stats) / max(len(stats), 1), 1)
    return jsonify({
        "total_movies": int(combined["movie_name"].nunique()),
        "total_reviews": total,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
        "neutral_pct": round(neu / total * 100, 1) if total else 0,
        "avg_watchability": avg_watch,
        "model_metrics": METRICS,
    })


@app.route("/api/movies", methods=["GET"])
def movies():
    err = dataset_guard()
    if err:
        return err
    return jsonify({"movies": current_movie_stats()})


@app.route("/api/movie/<path:movie_name>", methods=["GET"])
def movie_detail(movie_name):
    err = dataset_guard()
    if err:
        return err
    match = find_movie(movie_name)
    if not match:
        return jsonify({"error": "इस नाम की फिल्म अभी उपलब्ध नहीं है। आप लॉग इन करके इसकी पहली समीक्षा लिख सकते हैं।"}), 404

    combined = combined_dataframe()
    reviews = combined[combined["movie_name"].str.casefold() == movie_name.casefold()]
    sample_reviews = [
        {"text": row["raw_text"], "sentiment": row["sentiment"], "confidence": float(row["sentiment_confidence"])}
        for _, row in reviews.head(10).iterrows()
    ]
    user_reviews = user_reviews_dataframe()
    user_reviews = user_reviews[user_reviews["movie_name"].str.casefold() == movie_name.casefold()]
    result = dict(match)
    result["sample_reviews"] = sample_reviews
    result["user_reviews"] = [
        {"username": row["username"], "text": row["raw_text"], "sentiment": row["sentiment"], "confidence": float(row["sentiment_confidence"]), "created_at": row["created_at"]}
        for _, row in user_reviews.head(20).iterrows()
    ]
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    err = dataset_guard()
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    review = (payload.get("review") or "").strip()
    if not review:
        return jsonify({"error": "कृपया अपनी समीक्षा लिखें।"}), 400
    sentiment_en, confidence = predict_review(VECTORIZER, MODEL, review)
    hindi_map = {"positive": "सकारात्मक", "negative": "नकारात्मक", "neutral": "तटस्थ"}
    return jsonify({"sentiment": sentiment_en, "sentiment_hindi": hindi_map.get(sentiment_en, sentiment_en), "confidence": round(confidence, 4)})


@app.route("/api/reviews", methods=["POST"])
@login_required
def create_review():
    err = dataset_guard()
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    movie_name = (payload.get("movie_name") or "").strip()
    review = (payload.get("review") or "").strip()
    if not movie_name:
        return jsonify({"error": "कृपया फिल्म का नाम लिखें।"}), 400
    if not review:
        return jsonify({"error": "कृपया अपनी समीक्षा लिखें।"}), 400
    if len(movie_name) > 200:
        return jsonify({"error": "फिल्म का नाम बहुत लंबा है।"}), 400

    sentiment, confidence = predict_review(VECTORIZER, MODEL, review)
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn = db_connection()
    movie = conn.execute("SELECT id, movie_name FROM movies WHERE lower(movie_name) = lower(?)", (movie_name,)).fetchone()
    is_new_movie = movie is None
    if movie is None:
        cur = conn.execute("INSERT INTO movies(movie_name, created_at) VALUES (?, ?)", (movie_name, now))
        movie_id = cur.lastrowid
    else:
        movie_id = movie["id"]
    conn.execute(
        "INSERT INTO reviews(user_id, movie_id, review_text, sentiment, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session["user_id"], movie_id, review, sentiment, confidence, now),
    )
    conn.commit()
    conn.close()

    stats = find_movie(movie_name)
    hindi_map = {"positive": "सकारात्मक", "negative": "नकारात्मक", "neutral": "तटस्थ"}
    return jsonify({
        "success": True,
        "movie_name": movie_name,
        "is_new_movie": is_new_movie,
        "sentiment": sentiment,
        "sentiment_hindi": hindi_map[sentiment],
        "confidence": round(confidence, 4),
        "movie": stats,
        "message": "आपकी समीक्षा सफलतापूर्वक सहेजी और विश्लेषित की गई है।",
    }), 201


@app.route("/api/reviews/my", methods=["GET"])
@login_required
def my_reviews():
    conn = db_connection()
    rows = conn.execute("""
        SELECT r.id, m.movie_name, r.review_text, r.sentiment, r.confidence, r.created_at
        FROM reviews r JOIN movies m ON m.id = r.movie_id
        WHERE r.user_id = ? ORDER BY r.id DESC
    """, (session["user_id"],)).fetchall()
    conn.close()
    return jsonify({"reviews": [dict(r) for r in rows]})


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global VECTORIZER, MODEL, DATAFRAME, METRICS, LOAD_ERROR
    try:
        VECTORIZER, MODEL, DATAFRAME, METRICS = load_or_train(DATASET_PATH, force_retrain=True)
        seed_movie_names()
        LOAD_ERROR = None
        return jsonify({"status": "ok", "metrics": METRICS})
    except Exception as exc:
        LOAD_ERROR = str(exc)
        return jsonify({"error": "डेटासेट लोड नहीं हो सका। कृपया डेटासेट की जाँच करें।"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
