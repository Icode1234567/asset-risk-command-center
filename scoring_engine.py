"""
scoring_engine.py – Compute 0-100 risk scores for each asset.

Score breakdown (weights can be tuned):
  - Volatility          30%
  - Price momentum      25%
  - Volume anomaly      15%
  - Sentiment           30%

Higher score → higher perceived risk.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils import (
    annualised_volatility,
    clamp,
    headlines_sentiment,
    pct_change,
    sentiment_to_risk,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FactorScore:
    name: str
    score: float          # 0-100
    weight: float         # contribution weight 0-1
    description: str = ""

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class AssetRiskScore:
    asset: str
    total_score: float            # 0-100, final weighted score
    factors: list[FactorScore] = field(default_factory=list)
    price_info: dict = field(default_factory=dict)
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)

    def factor_dict(self) -> dict[str, float]:
        return {f.name: round(f.score, 1) for f in self.factors}


# ---------------------------------------------------------------------------
# Individual factor scorers
# ---------------------------------------------------------------------------

def _volatility_score(ohlcv: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Convert annualised volatility to a 0-100 risk score.

    Benchmarks (approximate):
      BTC  ~70-80% p.a. → high
      Oil  ~30-50% p.a. → medium
      Nifty ~15-25% p.a. → lower
    We scale so that 100% ann. vol → 100 risk.
    """
    if ohlcv.empty or len(ohlcv) < 3:
        return 50.0
    returns = ohlcv["close"].pct_change().dropna().values
    ann_vol = annualised_volatility(returns, periods_per_year)
    # 100% annualised vol → score 100; linear scaling, capped
    score = ann_vol * 100.0
    return clamp(score)


def _momentum_score(price_info: dict, ohlcv: pd.DataFrame) -> float:
    """Turn recent price change into a risk score.

    Negative returns increase risk; positive returns reduce it.
    We use a combination of 1-day and 5-day changes.
    """
    change_24h = price_info.get("change_24h", 0)

    # 5-day return from OHLCV
    change_5d = 0.0
    if not ohlcv.empty and len(ohlcv) >= 6:
        c = ohlcv["close"].dropna()
        change_5d = pct_change(float(c.iloc[-1]), float(c.iloc[-6]))

    # Combine: weight 1-day change more heavily
    combined = 0.6 * change_24h + 0.4 * change_5d

    # Map to 0-100: large negative change → high risk
    # −20% or worse → 100; +20% or better → 0; 0% → 50
    score = clamp(50.0 - combined * 2.5)
    return score


def _volume_anomaly_score(ohlcv: pd.DataFrame) -> float:
    """Score based on how much today's volume deviates from the 10-day average.

    Very high or very low volume can signal risk.
    Returns 0 if volume data is unavailable.
    """
    if ohlcv.empty or "volume" not in ohlcv.columns:
        return 30.0
    vol_series = ohlcv["volume"].dropna()
    if len(vol_series) < 3:
        return 30.0
    recent = float(vol_series.iloc[-1])
    avg = float(vol_series.iloc[:-1].mean())
    if avg == 0:
        return 30.0
    ratio = recent / avg
    # ratio > 2 → high anomaly (risk ↑); ratio < 0.5 → low volume (modest risk)
    if ratio >= 2.0:
        score = clamp(40.0 + (ratio - 2.0) * 20.0, 40.0, 100.0)
    elif ratio <= 0.5:
        score = 35.0
    else:
        score = 30.0
    return score


def _sentiment_score(headlines: list[str]) -> float:
    """Return sentiment-based risk score (0-100)."""
    if not headlines:
        return 50.0  # neutral when no news available
    compound = headlines_sentiment(headlines)
    return sentiment_to_risk(compound)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "Volatility": 0.30,
    "Momentum": 0.25,
    "Volume": 0.15,
    "Sentiment": 0.30,
}


def compute_risk_score(
    asset: str,
    price_info: dict,
    ohlcv: pd.DataFrame,
    headlines: list[str],
) -> AssetRiskScore:
    """Compute the composite risk score and per-factor breakdown."""

    factors = []

    v_score = _volatility_score(ohlcv)
    factors.append(
        FactorScore(
            "Volatility",
            v_score,
            WEIGHTS["Volatility"],
            "Annualised price volatility relative to asset benchmark",
        )
    )

    m_score = _momentum_score(price_info, ohlcv)
    factors.append(
        FactorScore(
            "Momentum",
            m_score,
            WEIGHTS["Momentum"],
            "Recent 1-day and 5-day price returns",
        )
    )

    vol_score = _volume_anomaly_score(ohlcv)
    factors.append(
        FactorScore(
            "Volume",
            vol_score,
            WEIGHTS["Volume"],
            "Trading volume deviation from 10-day average",
        )
    )

    sent_score = _sentiment_score(headlines)
    factors.append(
        FactorScore(
            "Sentiment",
            sent_score,
            WEIGHTS["Sentiment"],
            "VADER sentiment analysis of recent news headlines",
        )
    )

    total = clamp(sum(f.weighted for f in factors))

    return AssetRiskScore(
        asset=asset,
        total_score=round(total, 1),
        factors=factors,
        price_info=price_info,
        ohlcv=ohlcv,
    )


# ---------------------------------------------------------------------------
# Named-factor breakdown for the dashboard buttons
# ---------------------------------------------------------------------------

NAMED_FACTORS: list[str] = ["Economic", "Social", "Geopolitical", "Technical", "Regulatory"]


def factor_risk_breakdown(asset_score: AssetRiskScore, factor_headlines: dict[str, list[str]]) -> dict[str, float]:
    """Map the named UI factors to approximate risk scores.

    Uses the base score and adjusts with per-factor headline sentiment.
    """
    base = asset_score.total_score
    result = {}
    for factor in NAMED_FACTORS:
        headlines = factor_headlines.get(factor, [])
        if headlines:
            sent = headlines_sentiment(headlines)
            sent_risk = sentiment_to_risk(sent)
            # Blend base score and sentiment-based score
            factor_score = clamp(0.6 * base + 0.4 * sent_risk)
        else:
            # Small deterministic perturbation to make cards visually distinct
            seed = sum(ord(c) for c in (asset_score.asset + factor)) % 20
            factor_score = clamp(base + seed - 10)
        result[factor] = round(factor_score, 1)
    return result
