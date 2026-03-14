"""
factor_reports.py – Render detailed per-factor report pages in Streamlit.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

from utils import risk_label, clamp
from scoring_engine import AssetRiskScore, NAMED_FACTORS


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    _, color = risk_label(score)
    return color


def _gauge_chart(score: float, title: str) -> go.Figure:
    """Return a Plotly gauge figure for *score* [0-100]."""
    label, color = risk_label(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#888", "tickfont": {"color": "#888"}},
                "bar": {"color": color},
                "bgcolor": "#1A1C24",
                "bordercolor": "#333",
                "steps": [
                    {"range": [0, 30], "color": "#16423C"},
                    {"range": [30, 50], "color": "#2D4A22"},
                    {"range": [50, 75], "color": "#4A3000"},
                    {"range": [75, 100], "color": "#4A1010"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=260,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def _price_chart(ohlcv: pd.DataFrame, asset_name: str) -> go.Figure:
    """Candlestick + close-line chart from OHLCV DataFrame."""
    if ohlcv.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            title=dict(text="No price data available", font=dict(color="#888")),
            height=320,
        )
        return fig

    fig = go.Figure()
    if {"open", "high", "low", "close"}.issubset(ohlcv.columns):
        fig.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["close"],
                name="OHLC",
                increasing_line_color="#22C55E",
                decreasing_line_color="#EF4444",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"], mode="lines", name="Close"))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1C24",
        title=dict(text=f"{asset_name} – 30-Day Price", font=dict(color="#FAFAFA", size=14)),
        xaxis=dict(color="#888", gridcolor="#2A2C34", rangeslider=dict(visible=False)),
        yaxis=dict(color="#888", gridcolor="#2A2C34"),
        legend=dict(font=dict(color="#888")),
        height=320,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def _volatility_chart(ohlcv: pd.DataFrame, asset_name: str) -> go.Figure:
    """Rolling 7-day volatility chart."""
    if ohlcv.empty or len(ohlcv) < 5:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            title=dict(text="Insufficient data", font=dict(color="#888")),
            height=250,
        )
        return fig

    returns = ohlcv["close"].pct_change()
    rolling_vol = returns.rolling(7).std() * np.sqrt(252) * 100  # annualised %

    fig = go.Figure(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode="lines",
            line=dict(color="#FF4B4B", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.15)",
            name="Rolling 7-day Ann. Vol (%)",
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1C24",
        title=dict(text=f"{asset_name} – Rolling Volatility (Annualised %)", font=dict(color="#FAFAFA", size=14)),
        xaxis=dict(color="#888", gridcolor="#2A2C34"),
        yaxis=dict(color="#888", gridcolor="#2A2C34", title="Volatility %"),
        height=250,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def _volume_chart(ohlcv: pd.DataFrame, asset_name: str) -> go.Figure:
    """Bar chart of trading volume."""
    if ohlcv.empty or "volume" not in ohlcv.columns or ohlcv["volume"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            title=dict(text="Volume data not available", font=dict(color="#888")),
            height=220,
        )
        return fig

    vol = ohlcv["volume"].dropna()
    avg = vol.mean()
    colors = ["#22C55E" if v <= avg * 1.5 else "#EF4444" for v in vol]

    fig = go.Figure(go.Bar(x=vol.index, y=vol.values, marker_color=colors, name="Volume"))
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color="#EAB308",
        annotation_text="10-day avg",
        annotation_font_color="#EAB308",
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1C24",
        title=dict(text=f"{asset_name} – Trading Volume", font=dict(color="#FAFAFA", size=14)),
        xaxis=dict(color="#888", gridcolor="#2A2C34"),
        yaxis=dict(color="#888", gridcolor="#2A2C34"),
        height=220,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def _factor_bar_chart(factor_scores: dict[str, float], asset_name: str) -> go.Figure:
    """Horizontal bar chart of per-factor scores."""
    factors = list(factor_scores.keys())
    scores = [factor_scores[f] for f in factors]
    colors = [_score_color(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=factors,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.0f}" for s in scores],
            textposition="outside",
            textfont=dict(color="#FAFAFA"),
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1C24",
        title=dict(text=f"{asset_name} – Factor Risk Breakdown", font=dict(color="#FAFAFA", size=14)),
        xaxis=dict(range=[0, 110], color="#888", gridcolor="#2A2C34", title="Risk Score"),
        yaxis=dict(color="#FAFAFA"),
        height=280,
        margin=dict(t=40, b=20, l=100, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# News renderer
# ---------------------------------------------------------------------------

def _render_news(articles: list[dict], max_items: int = 6) -> None:
    """Render news articles as styled cards."""
    if not articles:
        st.info("No news articles available. Set NEWS_API_KEY in .env to enable live headlines.")
        return
    for art in articles[:max_items]:
        pub = art.get("publishedAt", "")
        if pub:
            try:
                pub = datetime.fromisoformat(pub.replace("Z", "+00:00")).strftime("%b %d, %Y %H:%M UTC")
            except Exception:
                pass
        source = art.get("source", "")
        title = art.get("title", "No title")
        desc = art.get("description", "")
        url = art.get("url", "#")
        st.markdown(
            f"""
            <div style="
                background:#1A1C24;
                border:1px solid #2A2C34;
                border-left:3px solid #FF4B4B;
                border-radius:6px;
                padding:12px 16px;
                margin-bottom:10px;
            ">
                <div style="font-size:0.75rem;color:#888;margin-bottom:4px;">{source} &nbsp;·&nbsp; {pub}</div>
                <a href="{url}" target="_blank" style="
                    color:#FAFAFA;font-size:0.95rem;font-weight:600;text-decoration:none;
                ">{title}</a>
                <div style="font-size:0.82rem;color:#AAA;margin-top:6px;">{desc or ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main entry-point: render a factor report
# ---------------------------------------------------------------------------

def render_factor_report(
    asset_score: AssetRiskScore,
    factor_name: str,
    factor_scores: dict[str, float],
    factor_articles: list[dict],
) -> None:
    """Render the full detailed page for one factor of one asset."""
    asset = asset_score.asset
    score = factor_scores.get(factor_name, asset_score.total_score)
    label, color = risk_label(score)

    # Header
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(90deg,#1A1C24,#0E1117);
            border:1px solid #2A2C34;
            border-left:4px solid {color};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:20px;
        ">
            <div style="font-size:0.8rem;color:#888;letter-spacing:1px;text-transform:uppercase;">
                {asset} · {factor_name} Factor Report
            </div>
            <div style="font-size:1.8rem;font-weight:700;color:#FAFAFA;margin-top:4px;">
                Risk Score: <span style="color:{color}">{score:.0f}</span>
                <span style="font-size:1rem;font-weight:400;color:{color};margin-left:8px;">{label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gauge + factor breakdown side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_gauge_chart(score, f"{factor_name} Risk"), use_container_width=True)
    with col2:
        st.plotly_chart(_factor_bar_chart(factor_scores, asset), use_container_width=True)

    # Price chart
    st.plotly_chart(_price_chart(asset_score.ohlcv, asset), use_container_width=True)

    # Volatility + Volume side by side
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(_volatility_chart(asset_score.ohlcv, asset), use_container_width=True)
    with col4:
        st.plotly_chart(_volume_chart(asset_score.ohlcv, asset), use_container_width=True)

    # News section
    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:600;color:#FAFAFA;margin:12px 0 8px;'>"
        f"📰 Latest {factor_name} News – {asset}</div>",
        unsafe_allow_html=True,
    )
    _render_news(factor_articles)

    # Factor-specific narrative
    _render_factor_narrative(asset, factor_name, score)


# ---------------------------------------------------------------------------
# Static narrative blurbs for each factor
# ---------------------------------------------------------------------------

_NARRATIVES: dict[str, dict[str, str]] = {
    "Economic": {
        "Bitcoin": (
            "Bitcoin's risk profile is closely tied to macro-economic conditions: "
            "inflation expectations, Federal Reserve rate decisions, and global liquidity cycles. "
            "Higher interest rates typically dampen speculative appetite, pushing risk higher."
        ),
        "Crude Oil": (
            "Crude Oil demand is driven by global GDP growth and industrial activity. "
            "Recessions or slowdowns in major economies (US, China, EU) can sharply reduce demand, "
            "increasing supply glut risk. Conversely, strong growth supports prices."
        ),
        "Nifty 50": (
            "The Nifty 50 reflects India's broader economic health—GDP growth, corporate earnings, "
            "inflation, and RBI policy. Rising interest rates or fiscal imbalances can elevate "
            "downside risk for equities."
        ),
    },
    "Social": {
        "Bitcoin": (
            "Social sentiment and retail participation are significant drivers of Bitcoin volatility. "
            "Social media trends, influencer commentary, and retail FOMO/panic cycles can amplify "
            "price moves dramatically."
        ),
        "Crude Oil": (
            "Public sentiment around energy prices affects consumer behaviour and political pressure "
            "on governments. High fuel prices can trigger demand-destruction and policy responses, "
            "introducing additional uncertainty."
        ),
        "Nifty 50": (
            "Retail investor participation in Indian markets has surged via platforms like Zerodha. "
            "Social media-driven narratives and domestic sentiment surveys are increasingly "
            "relevant risk indicators."
        ),
    },
    "Geopolitical": {
        "Bitcoin": (
            "Bitcoin faces geopolitical risk through potential government bans, capital controls, "
            "and sanctions regimes. Conflicts can both boost BTC (safe-haven narrative) and suppress "
            "it (regulatory crackdowns)."
        ),
        "Crude Oil": (
            "Geopolitical events in oil-producing regions (Middle East, Russia) are the primary "
            "driver of supply disruptions and price spikes. OPEC+ production decisions, sanctions, "
            "and military conflicts are key watchpoints."
        ),
        "Nifty 50": (
            "India's geopolitical relationships—particularly with China, Pakistan, and major trading "
            "partners—can affect foreign institutional investment flows and market sentiment. "
            "Border tensions and global trade wars introduce volatility."
        ),
    },
    "Technical": {
        "Bitcoin": (
            "Bitcoin's price action exhibits strong mean-reversion and trend-following behaviour. "
            "Key technical levels (200-day MA, Fibonacci retracements, RSI extremes) are widely "
            "watched by traders and often become self-fulfilling."
        ),
        "Crude Oil": (
            "Crude Oil futures trading is heavily influenced by technical positioning. "
            "Commitment-of-Traders (COT) data, support/resistance at round-number price levels, "
            "and seasonal patterns all play a role."
        ),
        "Nifty 50": (
            "The Nifty 50 has well-defined technical support and resistance zones. "
            "Institutional algo-trading, options open interest at key strikes, and index rebalancing "
            "events can create predictable short-term dynamics."
        ),
    },
    "Regulatory": {
        "Bitcoin": (
            "Regulatory risk is among the most binary risks for Bitcoin. SEC enforcement actions, "
            "CBDC competition, AML/KYC requirements, and potential outright bans in major "
            "jurisdictions remain significant tail risks."
        ),
        "Crude Oil": (
            "Energy transition policies, carbon taxes, environmental regulations, and trade sanctions "
            "are reshaping the long-term demand outlook for crude oil, introducing structural risk "
            "for producers and traders."
        ),
        "Nifty 50": (
            "SEBI regulations, corporate governance rules, FII investment limits, and changes to "
            "F&O margin requirements can affect market liquidity and valuations. "
            "Tax policy changes (LTCG, STT) also influence investor behaviour."
        ),
    },
}


def _render_factor_narrative(asset: str, factor: str, score: float) -> None:
    """Render a concise narrative paragraph for the factor."""
    text = _NARRATIVES.get(factor, {}).get(asset, "")
    if not text:
        return
    label, color = risk_label(score)
    st.markdown(
        f"""
        <div style="
            background:#1A1C24;
            border:1px solid #2A2C34;
            border-radius:8px;
            padding:16px 20px;
            margin-top:16px;
        ">
            <div style="font-size:0.85rem;font-weight:600;color:{color};margin-bottom:8px;">
                {factor} Risk Analysis · {asset}
            </div>
            <div style="font-size:0.9rem;color:#CCC;line-height:1.6;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
