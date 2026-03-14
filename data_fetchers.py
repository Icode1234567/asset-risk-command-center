"""
data_fetchers.py – Fetch OHLCV and news data for Bitcoin, Crude Oil, and Nifty 50.

All public API calls use free / no-key-required endpoints where possible.
NewsAPI headlines require NEWS_API_KEY in .env (gracefully degraded otherwise).
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from utils import cached

load_dotenv()
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
NEWSAPI_BASE = "https://newsapi.org/v2"

# ---------------------------------------------------------------------------
# Bitcoin  (CoinGecko – no key required for basic calls)
# ---------------------------------------------------------------------------

@cached(ttl_seconds=300)
def fetch_bitcoin_ohlcv(days: int = 30) -> pd.DataFrame:
    """Return daily OHLCV data for Bitcoin from CoinGecko."""
    url = f"{COINGECKO_BASE}/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        # Synthesise a volume proxy (not provided by OHLC endpoint)
        df["volume"] = np.nan
        return df
    except Exception as exc:
        logger.warning("Bitcoin OHLCV fetch failed: %s", exc)
        return _empty_ohlcv()


@cached(ttl_seconds=300)
def fetch_bitcoin_price() -> dict:
    """Return latest Bitcoin price info."""
    url = f"{COINGECKO_BASE}/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
        "include_market_cap": "true",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("bitcoin", {})
        return {
            "price": data.get("usd", 0),
            "change_24h": data.get("usd_24h_change", 0),
            "volume_24h": data.get("usd_24h_vol", 0),
            "market_cap": data.get("usd_market_cap", 0),
        }
    except Exception as exc:
        logger.warning("Bitcoin price fetch failed: %s", exc)
        return {"price": 0, "change_24h": 0, "volume_24h": 0, "market_cap": 0}


# ---------------------------------------------------------------------------
# Crude Oil  (yfinance – CL=F futures)
# ---------------------------------------------------------------------------

@cached(ttl_seconds=300)
def fetch_crude_oil_ohlcv(days: int = 30) -> pd.DataFrame:
    """Return daily OHLCV data for WTI Crude Oil futures (CL=F)."""
    try:
        import yfinance as yf
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 5)  # buffer for weekends
        ticker = yf.Ticker("CL=F")
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            return _empty_ohlcv()
        df.index = pd.to_datetime(df.index).tz_convert(None)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].tail(days)
    except Exception as exc:
        logger.warning("Crude Oil OHLCV fetch failed: %s", exc)
        return _empty_ohlcv()


@cached(ttl_seconds=300)
def fetch_crude_oil_price() -> dict:
    """Return latest WTI Crude Oil price info."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("CL=F")
        info = ticker.fast_info
        hist = ticker.history(period="2d")
        price = float(info.last_price) if hasattr(info, "last_price") else 0
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change_24h = (price - prev_close) / prev_close * 100 if prev_close else 0
        volume = float(hist["Volume"].iloc[-1]) if not hist.empty else 0
        return {"price": price, "change_24h": change_24h, "volume_24h": volume}
    except Exception as exc:
        logger.warning("Crude Oil price fetch failed: %s", exc)
        return {"price": 0, "change_24h": 0, "volume_24h": 0}


# ---------------------------------------------------------------------------
# Nifty 50  (yfinance – ^NSEI)
# ---------------------------------------------------------------------------

@cached(ttl_seconds=300)
def fetch_nifty_ohlcv(days: int = 30) -> pd.DataFrame:
    """Return daily OHLCV data for Nifty 50 (^NSEI)."""
    try:
        import yfinance as yf
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 10)
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            return _empty_ohlcv()
        df.index = pd.to_datetime(df.index).tz_convert(None)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].tail(days)
    except Exception as exc:
        logger.warning("Nifty 50 OHLCV fetch failed: %s", exc)
        return _empty_ohlcv()


@cached(ttl_seconds=300)
def fetch_nifty_price() -> dict:
    """Return latest Nifty 50 price info."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="2d")
        if hist.empty:
            return {"price": 0, "change_24h": 0, "volume_24h": 0}
        price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change_24h = (price - prev_close) / prev_close * 100 if prev_close else 0
        volume = float(hist["Volume"].iloc[-1])
        return {"price": price, "change_24h": change_24h, "volume_24h": volume}
    except Exception as exc:
        logger.warning("Nifty price fetch failed: %s", exc)
        return {"price": 0, "change_24h": 0, "volume_24h": 0}


# ---------------------------------------------------------------------------
# News headlines  (NewsAPI – requires NEWS_API_KEY)
# ---------------------------------------------------------------------------

@cached(ttl_seconds=600)
def fetch_news_headlines(query: str, page_size: int = 10) -> list[dict]:
    """Fetch top news headlines for *query*.

    Returns a list of dicts with keys: title, description, url, publishedAt.
    Falls back to an empty list when NEWS_API_KEY is not set.
    """
    if not NEWS_API_KEY:
        logger.info("NEWS_API_KEY not set – returning empty headlines for '%s'.", query)
        return []
    url = f"{NEWSAPI_BASE}/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", "#"),
                "publishedAt": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as exc:
        logger.warning("News fetch failed for '%s': %s", query, exc)
        return []


# ---------------------------------------------------------------------------
# Asset news query map
# ---------------------------------------------------------------------------

ASSET_NEWS_QUERIES: dict[str, list[str]] = {
    "Bitcoin": ["bitcoin", "crypto regulation", "crypto market"],
    "Crude Oil": ["crude oil price", "OPEC", "oil supply"],
    "Nifty 50": ["nifty 50", "Indian stock market", "NSE India"],
}

ASSET_FACTOR_NEWS_QUERIES: dict[str, dict[str, str]] = {
    "Bitcoin": {
        "Economic": "bitcoin economy inflation fed",
        "Social": "bitcoin social media sentiment retail",
        "Geopolitical": "bitcoin geopolitical sanctions crypto ban",
        "Technical": "bitcoin technical analysis price chart",
        "Regulatory": "bitcoin regulation SEC crypto law",
    },
    "Crude Oil": {
        "Economic": "crude oil demand GDP economy",
        "Social": "oil energy crisis fuel prices public",
        "Geopolitical": "oil geopolitical OPEC Middle East conflict",
        "Technical": "crude oil technical analysis futures",
        "Regulatory": "oil regulation carbon tax energy policy",
    },
    "Nifty 50": {
        "Economic": "India economy GDP inflation RBI",
        "Social": "India market retail investor sentiment",
        "Geopolitical": "India geopolitical trade war border",
        "Technical": "Nifty 50 technical analysis chart support",
        "Regulatory": "SEBI India stock market regulation",
    },
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
