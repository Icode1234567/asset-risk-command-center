"""
utils.py – Helper functions for caching, sentiment analysis, and formatting.
"""

import os
import time
import functools
import logging
from typing import Any, Callable, Optional

import nltk
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VADER sentiment – download once at import time
# ---------------------------------------------------------------------------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception as exc:
    logger.warning("VADER not available: %s", exc)
    _sia = None
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simple in-process time-based cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[Any, float]] = {}


def cached(ttl_seconds: int = 300) -> Callable:
    """Decorator: cache function results for *ttl_seconds* seconds."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            now = time.time()
            if key in _cache:
                value, ts = _cache[key]
                if now - ts < ttl_seconds:
                    return value
            result = func(*args, **kwargs)
            _cache[key] = (result, now)
            return result

        return wrapper

    return decorator


def clear_cache() -> None:
    """Remove all cached entries."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def sentiment_score(text: str) -> float:
    """Return a compound sentiment score in [-1, 1] using VADER.

    Falls back to 0.0 if VADER is unavailable.
    """
    if not VADER_AVAILABLE or not text:
        return 0.0
    return _sia.polarity_scores(str(text))["compound"]


def headlines_sentiment(headlines: list[str]) -> float:
    """Return the mean VADER compound score across a list of headlines."""
    if not headlines:
        return 0.0
    scores = [sentiment_score(h) for h in headlines if h]
    return float(np.mean(scores)) if scores else 0.0


def sentiment_to_risk(sentiment: float) -> float:
    """Convert VADER compound score [-1, 1] to a risk contribution [0, 100].

    Most-negative sentiment → highest risk (100).
    Most-positive sentiment → lowest risk (0).
    """
    # Invert: -1 → 100, 0 → 50, +1 → 0
    return round((1.0 - sentiment) / 2.0 * 100, 2)


# ---------------------------------------------------------------------------
# Number / display helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def risk_label(score: float) -> tuple[str, str]:
    """Return a human-readable label and hex colour for a risk score."""
    if score >= 75:
        return "High Risk", "#EF4444"
    if score >= 50:
        return "Medium-High Risk", "#F97316"
    if score >= 30:
        return "Medium Risk", "#EAB308"
    return "Low Risk", "#22C55E"


def pct_change(current: float, previous: float) -> float:
    """Percentage change from *previous* to *current*."""
    if previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100


def annualised_volatility(returns: "np.ndarray", periods_per_year: int = 252) -> float:
    """Annualised volatility from a 1-D array of period returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
