"""
recommendation.py
------------------
Movie-level aggregation, watchability score, and Hindi recommendation
categories.
"""
import pandas as pd

RECOMMENDATION_BANDS = [
    (80, 101, "🔥", "ज़रूर देखें",
     "दर्शकों की प्रतिक्रियाएँ काफी सकारात्मक हैं। यह फिल्म देखने के लिए एक अच्छी पसंद हो सकती है।"),
    (65, 80, "👍", "देखने की सलाह",
     "फिल्म को लेकर प्रतिक्रियाएँ सामान्यतः सकारात्मक हैं। आप इसे अपनी पसंद के अनुसार देख सकते हैं।"),
    (50, 65, "😐", "एक बार देख सकते हैं",
     "प्रतिक्रियाएँ मिली-जुली हैं। फिल्म आपकी व्यक्तिगत पसंद पर निर्भर कर सकती है।"),
    (30, 50, "👎", "सोच-समझकर देखें",
     "फिल्म को लेकर नकारात्मक प्रतिक्रियाएँ अपेक्षाकृत अधिक हैं।"),
    (0, 30, "❌", "न देखने की सलाह",
     "समीक्षाओं में नकारात्मक भावनाएँ काफी अधिक हैं।"),
]


def recommendation_for_score(score: float) -> dict:
    for lo, hi, emoji, label, desc in RECOMMENDATION_BANDS:
        if lo <= score < hi:
            return {"emoji": emoji, "label": label, "description": desc}
    return RECOMMENDATION_BANDS[-1]


def compute_watchability(pos: int, neu: int, neg: int, ratings: "pd.Series|None" = None) -> float:
    """
    Sentiment-based score in [0, 100]:
        positive -> +1, neutral -> 0, negative -> -1, averaged and rescaled.
    If a numeric rating column is available, it is blended in as a
    secondary factor (70% sentiment / 30% normalized rating).
    """
    n = pos + neu + neg
    if n == 0:
        return 0.0
    raw = (pos * 1 + neu * 0 + neg * -1) / n  # -1..1
    sentiment_score = (raw + 1) / 2 * 100

    if ratings is not None and ratings.notna().any():
        rmin, rmax = ratings.min(), ratings.max()
        if rmax > rmin:
            rating_score = ((ratings.mean() - rmin) / (rmax - rmin)) * 100
        else:
            rating_score = 50.0
        return round(sentiment_score * 0.7 + rating_score * 0.3, 1)

    return round(sentiment_score, 1)


def build_movie_stats(df: pd.DataFrame) -> list:
    stats = []
    for movie_name, g in df.groupby("movie_name"):
        n = len(g)
        pos = int((g["sentiment"] == "positive").sum())
        neg = int((g["sentiment"] == "negative").sum())
        neu = int((g["sentiment"] == "neutral").sum())
        ratings = g["rating"] if "rating" in g.columns else None
        score = compute_watchability(pos, neu, neg, ratings)
        rec = recommendation_for_score(score)
        stats.append({
            "movie_name": movie_name,
            "total_reviews": n,
            "positive_pct": round(pos / n * 100, 1),
            "negative_pct": round(neg / n * 100, 1),
            "neutral_pct": round(neu / n * 100, 1),
            "watchability_score": score,
            "recommendation": rec,
        })
    stats.sort(key=lambda r: r["watchability_score"], reverse=True)
    return stats
