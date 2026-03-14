"""
demo_data.py – Generate realistic mock data for demonstration / offline use.

Called automatically when live API data is unavailable (price == 0).
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


_RNG = np.random.default_rng(seed=42)


def _gbm_prices(start_price: float, mu: float, sigma: float, n: int = 30) -> np.ndarray:
    """Geometric Brownian Motion price series."""
    dt = 1 / 252
    returns = _RNG.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n)
    prices = start_price * np.exp(np.cumsum(returns))
    return prices


def demo_bitcoin_price() -> dict:
    prices = _gbm_prices(68_000, mu=0.3, sigma=0.7, n=2)
    price = float(prices[-1])
    prev = float(prices[-2])
    change = (price - prev) / prev * 100
    return {
        "price": round(price, 2),
        "change_24h": round(change, 2),
        "volume_24h": round(float(_RNG.uniform(25e9, 45e9)), 0),
        "market_cap": round(price * 19_500_000, 0),
    }


def demo_bitcoin_ohlcv(days: int = 30) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=days, freq="D")
    close = _gbm_prices(65_000, mu=0.2, sigma=0.75, n=days)
    hi = close * _RNG.uniform(1.005, 1.04, days)
    lo = close * _RNG.uniform(0.96, 0.995, days)
    op = lo + _RNG.uniform(0, 1, days) * (hi - lo)
    vol = _RNG.uniform(20e9, 50e9, days)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=dates)


def demo_crude_oil_price() -> dict:
    prices = _gbm_prices(78.0, mu=0.05, sigma=0.35, n=2)
    price = float(prices[-1])
    prev = float(prices[-2])
    change = (price - prev) / prev * 100
    return {
        "price": round(price, 2),
        "change_24h": round(change, 2),
        "volume_24h": round(float(_RNG.uniform(300_000, 700_000)), 0),
    }


def demo_crude_oil_ohlcv(days: int = 30) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=days, freq="B")[:days]
    close = _gbm_prices(78.0, mu=0.04, sigma=0.30, n=days)
    hi = close * _RNG.uniform(1.003, 1.02, days)
    lo = close * _RNG.uniform(0.98, 0.997, days)
    op = lo + _RNG.uniform(0, 1, days) * (hi - lo)
    vol = _RNG.uniform(200_000, 800_000, days)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=dates)


def demo_nifty_price() -> dict:
    prices = _gbm_prices(22_000, mu=0.12, sigma=0.18, n=2)
    price = float(prices[-1])
    prev = float(prices[-2])
    change = (price - prev) / prev * 100
    return {
        "price": round(price, 2),
        "change_24h": round(change, 2),
        "volume_24h": round(float(_RNG.uniform(5e8, 15e8)), 0),
    }


def demo_nifty_ohlcv(days: int = 30) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    dates = pd.date_range(end=end, periods=days, freq="B")[:days]
    close = _gbm_prices(22_000, mu=0.10, sigma=0.16, n=days)
    hi = close * _RNG.uniform(1.002, 1.015, days)
    lo = close * _RNG.uniform(0.985, 0.998, days)
    op = lo + _RNG.uniform(0, 1, days) * (hi - lo)
    vol = _RNG.uniform(3e8, 1.2e9, days)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=dates)


def demo_headlines(asset: str, factor: str = "") -> list[dict]:
    """Return realistic-looking demo headlines for *asset*."""
    _headlines: dict[str, list[str]] = {
        "Bitcoin": [
            "Bitcoin climbs past $70K as institutional demand surges",
            "Fed rate decision clouds crypto outlook for Q2",
            "Crypto regulation bill advances in US Senate",
            "Bitcoin ETF inflows hit record $500M in single day",
            "Mining difficulty reaches all-time high amid hashrate growth",
            "Analysts warn of short-term correction after parabolic run",
        ],
        "Crude Oil": [
            "OPEC+ extends production cuts through end of year",
            "US crude inventories fall sharply, tightening global supply",
            "China demand recovery boosts oil market sentiment",
            "Middle East tensions drive safe-haven bids in energy",
            "IEA raises 2025 demand forecast citing stronger growth",
            "Russia redirects oil exports as Western sanctions bite",
        ],
        "Nifty 50": [
            "Nifty 50 surges 1.2% on strong Q3 earnings season",
            "FII inflows return as India GDP growth surprises",
            "RBI holds rates steady, signals accommodative stance",
            "SEBI tightens F&O margin requirements for retail traders",
            "India inflation dips to 4.5%, bolstering rate-cut hopes",
            "Tech sector leads Nifty rally on global risk appetite",
        ],
    }
    titles = _headlines.get(asset, [])
    return [
        {
            "title": t,
            "description": "",
            "url": "#",
            "publishedAt": (datetime.now(timezone.utc) - timedelta(hours=i * 4)).isoformat() + "Z",
            "source": "Demo Data",
        }
        for i, t in enumerate(titles)
    ]
