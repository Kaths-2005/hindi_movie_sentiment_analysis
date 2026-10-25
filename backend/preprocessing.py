"""
preprocessing.py
-----------------
Text cleaning utilities for Hindi movie reviews.
Used both at training time (model.py) and at prediction time (app.py),
so a live review typed by a user is cleaned the exact same way as the
reviews the model was trained on.
"""
import re
import pandas as pd

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

# A small, extensible Hindi stopword list (kept short on purpose --
# aggressive stopword removal hurts an already-small dataset).
HINDI_STOPWORDS = {
    "है", "हैं", "था", "थी", "थे", "और", "का", "की", "के", "को", "में",
    "से", "पर", "यह", "वह", "ये", "वो", "एक", "भी", "तो", "ही", "हो",
    "कि", "जो", "इस", "उस", "अपने", "क्या", "कर", "रहा", "रही", "रहे",
}


def extract_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(EMOJI_PATTERN.findall(text))


def clean_text(text: str) -> str:
    """Remove emojis, punctuation, digits, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"[.,!?;:\"'()\[\]{}।\-_/\\|*#@%^&+=~`<>0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return clean_text(text).split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in HINDI_STOPWORDS]


def preprocess_for_model(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> rejoin.
    This is the exact text representation the TF-IDF vectorizer expects.
    """
    tokens = remove_stopwords(tokenize(text))
    return " ".join(tokens)


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    """Load a raw CSV/XLSX review export and normalize it into a
    standard schema: movie_name, raw_text, emojis, clean_text.
    Detects likely column names instead of assuming a fixed schema.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    def find_col(candidates):
        for c in df.columns:
            if c.lower() in candidates:
                return c
        return None

    movie_col = find_col({"movie_name", "movie", "title", "film"})
    text_col = find_col({"raw_text", "review", "review_text", "text", "comment", "final_preprocessed_text"})
    rating_col = find_col({"rating", "stars", "score"})

    if movie_col is None or text_col is None:
        raise ValueError("डेटासेट में फिल्म नाम या समीक्षा टेक्स्ट कॉलम नहीं मिला।")

    out = pd.DataFrame()
    out["movie_name"] = df[movie_col].fillna("Unknown").astype(str).str.strip()
    out["raw_text"] = df[text_col].astype(str)
    if rating_col:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")

    # drop empty / duplicate reviews
    out = out[out["raw_text"].str.strip() != ""]
    out = out.drop_duplicates(subset=["movie_name", "raw_text"])
    out = out[out["movie_name"].str.strip() != ""]

    out["emojis_extracted"] = out["raw_text"].apply(extract_emojis)
    out["final_preprocessed_text"] = out["raw_text"].apply(preprocess_for_model)
    out = out[out["final_preprocessed_text"].str.strip() != ""]
    out = out.reset_index(drop=True)
    out["review_id"] = out.index + 1
    return out
