"""
app.py – Asset Risk Command Center
Main Streamlit application entry-point.

Run with:
    streamlit run app.py
"""

import time
import streamlit as st
import plotly.graph_objects as go

from data_fetchers import (
    fetch_bitcoin_ohlcv,
    fetch_bitcoin_price,
    fetch_crude_oil_ohlcv,
    fetch_crude_oil_price,
    fetch_nifty_ohlcv,
    fetch_nifty_price,
    fetch_news_headlines,
    ASSET_NEWS_QUERIES,
    ASSET_FACTOR_NEWS_QUERIES,
)
from scoring_engine import (
    compute_risk_score,
    factor_risk_breakdown,
    AssetRiskScore,
    NAMED_FACTORS,
)
from factor_reports import render_factor_report
from utils import risk_label, clear_cache
import demo_data as _demo

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Asset Risk Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS – dark theme polish
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}

    /* Body / global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1A1C24;
        border: 1px solid #2A2C34;
        border-radius: 8px;
        padding: 10px 14px;
    }

    /* Factor buttons */
    .stButton > button {
        background: #1A1C24;
        border: 1px solid #2A2C34;
        color: #FAFAFA;
        border-radius: 6px;
        font-size: 0.82rem;
        padding: 6px 14px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        border-color: #FF4B4B;
        color: #FF4B4B;
        background: #1F1215;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(255,75,75,0.4);
    }

    /* Headers */
    h1, h2, h3 { color: #FAFAFA !important; }

    /* Divider */
    hr { border-color: #2A2C34; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1A1C24;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #FAFAFA !important;
        border-bottom: 2px solid #FF4B4B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "page": "dashboard",      # "dashboard" | "factor_report"
        "selected_asset": None,
        "selected_factor": None,
        "last_refresh": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Data loading  (cached inside data_fetchers)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_all_data() -> dict:
    """Load all asset data in one go; cached for 5 minutes.

    Falls back to deterministic demo data when live APIs are unreachable
    (price == 0 is treated as a failed fetch).
    """
    assets: dict[str, dict] = {}

    # --- Bitcoin ---
    btc_price = fetch_bitcoin_price()
    btc_ohlcv = fetch_bitcoin_ohlcv(days=30)
    # Fallback to demo data when network is unavailable
    if btc_price.get("price", 0) == 0:
        btc_price = _demo.demo_bitcoin_price()
        btc_ohlcv = _demo.demo_bitcoin_ohlcv(30)
    btc_headlines_raw = []
    for q in ASSET_NEWS_QUERIES["Bitcoin"]:
        btc_headlines_raw += fetch_news_headlines(q, page_size=5)
    if not btc_headlines_raw:
        btc_headlines_raw = _demo.demo_headlines("Bitcoin")
    btc_headline_texts = [a["title"] for a in btc_headlines_raw]
    btc_score = compute_risk_score("Bitcoin", btc_price, btc_ohlcv, btc_headline_texts)

    btc_factor_hl: dict[str, list] = {}
    btc_factor_articles: dict[str, list] = {}
    for factor, query in ASSET_FACTOR_NEWS_QUERIES["Bitcoin"].items():
        arts = fetch_news_headlines(query, page_size=8)
        if not arts:
            arts = _demo.demo_headlines("Bitcoin", factor)
        btc_factor_hl[factor] = [a["title"] for a in arts]
        btc_factor_articles[factor] = arts

    btc_factor_scores = factor_risk_breakdown(btc_score, btc_factor_hl)
    assets["Bitcoin"] = {
        "score": btc_score,
        "factor_scores": btc_factor_scores,
        "factor_articles": btc_factor_articles,
        "price_info": btc_price,
    }

    # --- Crude Oil ---
    oil_price = fetch_crude_oil_price()
    oil_ohlcv = fetch_crude_oil_ohlcv(days=30)
    if oil_price.get("price", 0) == 0:
        oil_price = _demo.demo_crude_oil_price()
        oil_ohlcv = _demo.demo_crude_oil_ohlcv(30)
    oil_headlines_raw = []
    for q in ASSET_NEWS_QUERIES["Crude Oil"]:
        oil_headlines_raw += fetch_news_headlines(q, page_size=5)
    if not oil_headlines_raw:
        oil_headlines_raw = _demo.demo_headlines("Crude Oil")
    oil_headline_texts = [a["title"] for a in oil_headlines_raw]
    oil_score = compute_risk_score("Crude Oil", oil_price, oil_ohlcv, oil_headline_texts)

    oil_factor_hl: dict[str, list] = {}
    oil_factor_articles: dict[str, list] = {}
    for factor, query in ASSET_FACTOR_NEWS_QUERIES["Crude Oil"].items():
        arts = fetch_news_headlines(query, page_size=8)
        if not arts:
            arts = _demo.demo_headlines("Crude Oil", factor)
        oil_factor_hl[factor] = [a["title"] for a in arts]
        oil_factor_articles[factor] = arts

    oil_factor_scores = factor_risk_breakdown(oil_score, oil_factor_hl)
    assets["Crude Oil"] = {
        "score": oil_score,
        "factor_scores": oil_factor_scores,
        "factor_articles": oil_factor_articles,
        "price_info": oil_price,
    }

    # --- Nifty 50 ---
    nifty_price = fetch_nifty_price()
    nifty_ohlcv = fetch_nifty_ohlcv(days=30)
    if nifty_price.get("price", 0) == 0:
        nifty_price = _demo.demo_nifty_price()
        nifty_ohlcv = _demo.demo_nifty_ohlcv(30)
    nifty_headlines_raw = []
    for q in ASSET_NEWS_QUERIES["Nifty 50"]:
        nifty_headlines_raw += fetch_news_headlines(q, page_size=5)
    if not nifty_headlines_raw:
        nifty_headlines_raw = _demo.demo_headlines("Nifty 50")
    nifty_headline_texts = [a["title"] for a in nifty_headlines_raw]
    nifty_score = compute_risk_score("Nifty 50", nifty_price, nifty_ohlcv, nifty_headline_texts)

    nifty_factor_hl: dict[str, list] = {}
    nifty_factor_articles: dict[str, list] = {}
    for factor, query in ASSET_FACTOR_NEWS_QUERIES["Nifty 50"].items():
        arts = fetch_news_headlines(query, page_size=8)
        if not arts:
            arts = _demo.demo_headlines("Nifty 50", factor)
        nifty_factor_hl[factor] = [a["title"] for a in arts]
        nifty_factor_articles[factor] = arts

    nifty_factor_scores = factor_risk_breakdown(nifty_score, nifty_factor_hl)
    assets["Nifty 50"] = {
        "score": nifty_score,
        "factor_scores": nifty_factor_scores,
        "factor_articles": nifty_factor_articles,
        "price_info": nifty_price,
    }

    return assets


# ---------------------------------------------------------------------------
# Gauge chart helper
# ---------------------------------------------------------------------------

def _mini_gauge(score: float, asset: str) -> go.Figure:
    """Compact gauge for the asset dashboard card."""
    label, color = risk_label(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 36, "color": color}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickcolor": "#555",
                    "tickfont": {"color": "#555", "size": 10},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#1A1C24",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#16423C"},
                    {"range": [30, 50], "color": "#2D4A22"},
                    {"range": [50, 75], "color": "#4A3000"},
                    {"range": [75, 100], "color": "#4A1010"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=200,
        margin=dict(t=20, b=10, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Asset card renderer
# ---------------------------------------------------------------------------

ASSET_META: dict[str, dict] = {
    "Bitcoin": {"icon": "₿", "unit": "USD", "color": "#F7931A"},
    "Crude Oil": {"icon": "🛢️", "unit": "USD/bbl", "color": "#8B6914"},
    "Nifty 50": {"icon": "📈", "unit": "INR", "color": "#00A86B"},
}


def _fmt_price(price: float, asset: str) -> str:
    if asset == "Nifty 50":
        return f"₹{price:,.2f}"
    if asset == "Bitcoin":
        return f"${price:,.0f}"
    return f"${price:,.2f}"


def _render_asset_card(asset: str, data: dict) -> None:
    """Render the full asset card including gauge and factor buttons."""
    asset_score: AssetRiskScore = data["score"]
    factor_scores: dict[str, float] = data["factor_scores"]
    price_info: dict = data["price_info"]
    meta = ASSET_META[asset]
    total = asset_score.total_score
    label, color = risk_label(total)

    # Card wrapper
    st.markdown(
        f"""
        <div style="
            background:#1A1C24;
            border:1px solid #2A2C34;
            border-top:3px solid {meta['color']};
            border-radius:10px;
            padding:18px 20px 14px;
            margin-bottom:6px;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:1.6rem;">{meta['icon']}</span>
                    <span style="font-size:1.25rem;font-weight:700;color:#FAFAFA;margin-left:8px;">{asset}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.4rem;font-weight:700;color:#FAFAFA;">
                        {_fmt_price(price_info.get("price", 0), asset)}
                    </div>
                    <div style="font-size:0.85rem;color:{'#22C55E' if price_info.get('change_24h',0)>=0 else '#EF4444'};">
                        {'▲' if price_info.get('change_24h',0)>=0 else '▼'}
                        {abs(price_info.get('change_24h', 0)):.2f}% (24h)
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gauge
    st.plotly_chart(_mini_gauge(total, asset), use_container_width=True, key=f"gauge_{asset}")

    # Risk label badge
    st.markdown(
        f"""
        <div style="text-align:center;margin:-8px 0 10px;">
            <span style="
                background:{color}22;
                border:1px solid {color};
                color:{color};
                border-radius:20px;
                padding:3px 14px;
                font-size:0.8rem;
                font-weight:600;
                letter-spacing:0.5px;
            ">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Factor breakdown mini-scores
    st.markdown(
        "<div style='font-size:0.75rem;color:#888;letter-spacing:0.5px;margin-bottom:6px;'>"
        "RISK FACTORS</div>",
        unsafe_allow_html=True,
    )
    btn_cols = st.columns(len(NAMED_FACTORS))
    for i, factor in enumerate(NAMED_FACTORS):
        f_score = factor_scores.get(factor, total)
        _, f_color = risk_label(f_score)
        with btn_cols[i]:
            st.markdown(
                f"""
                <div style="text-align:center;font-size:0.65rem;color:{f_color};margin-bottom:2px;">
                    {f_score:.0f}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(factor, key=f"btn_{asset}_{factor}"):
                st.session_state["page"] = "factor_report"
                st.session_state["selected_asset"] = asset
                st.session_state["selected_factor"] = factor
                st.rerun()


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def _render_dashboard(all_data: dict) -> None:
    # Title bar
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown(
            """
            <h1 style="font-size:1.8rem;margin-bottom:0;">
                📊 Asset Risk Command Center
            </h1>
            <div style="font-size:0.85rem;color:#888;margin-top:2px;">
                Real-time risk scores for Bitcoin, Crude Oil, and Nifty 50
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.cache_data.clear()
            clear_cache()
            st.rerun()

    st.markdown("<hr style='border-color:#2A2C34;margin:12px 0 20px;'>", unsafe_allow_html=True)

    # Overall risk summary row
    scores = {a: d["score"].total_score for a, d in all_data.items()}
    avg_score = sum(scores.values()) / len(scores)
    avg_label, avg_color = risk_label(avg_score)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Portfolio Risk (Avg)", f"{avg_score:.1f} / 100", delta=None)
    with m2:
        btc_ch = all_data["Bitcoin"]["price_info"].get("change_24h", 0)
        st.metric("Bitcoin 24h", f"${all_data['Bitcoin']['price_info'].get('price',0):,.0f}",
                  delta=f"{btc_ch:+.2f}%")
    with m3:
        oil_ch = all_data["Crude Oil"]["price_info"].get("change_24h", 0)
        st.metric("Crude Oil 24h", f"${all_data['Crude Oil']['price_info'].get('price',0):,.2f}",
                  delta=f"{oil_ch:+.2f}%")
    with m4:
        nifty_ch = all_data["Nifty 50"]["price_info"].get("change_24h", 0)
        st.metric("Nifty 50 24h", f"₹{all_data['Nifty 50']['price_info'].get('price',0):,.0f}",
                  delta=f"{nifty_ch:+.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Asset cards
    card_cols = st.columns(3)
    for idx, asset in enumerate(["Bitcoin", "Crude Oil", "Nifty 50"]):
        with card_cols[idx]:
            _render_asset_card(asset, all_data[asset])

    # Comparative heatmap table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;color:#FAFAFA;margin-bottom:10px;'>"
        "Risk Factor Heatmap</div>",
        unsafe_allow_html=True,
    )
    _render_heatmap(all_data)

    # Footer
    st.markdown(
        """
        <hr style='border-color:#2A2C34;margin-top:30px;'>
        <div style='font-size:0.75rem;color:#555;text-align:center;'>
            Data refreshes every 5 minutes · Prices from CoinGecko &amp; Yahoo Finance
            · Scores are informational only, not financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_heatmap(all_data: dict) -> None:
    """Render a colour-coded factor risk heatmap table."""
    import plotly.figure_factory as ff
    import numpy as np

    assets = ["Bitcoin", "Crude Oil", "Nifty 50"]
    factor_matrix = []
    for asset in assets:
        row = [all_data[asset]["factor_scores"].get(f, 50) for f in NAMED_FACTORS]
        factor_matrix.append(row)

    z = np.array(factor_matrix, dtype=float)

    # Heatmap
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=NAMED_FACTORS,
            y=assets,
            colorscale=[
                [0.0, "#16423C"],
                [0.3, "#22C55E"],
                [0.5, "#EAB308"],
                [0.75, "#F97316"],
                [1.0, "#EF4444"],
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#888"),
                title=dict(text="Risk", font=dict(color="#888")),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1C24",
        xaxis=dict(color="#FAFAFA", tickfont=dict(color="#FAFAFA")),
        yaxis=dict(color="#FAFAFA", tickfont=dict(color="#FAFAFA")),
        height=220,
        margin=dict(t=10, b=10, l=80, r=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="heatmap")


# ---------------------------------------------------------------------------
# Factor report page
# ---------------------------------------------------------------------------

def _render_factor_report_page(all_data: dict) -> None:
    asset = st.session_state["selected_asset"]
    factor = st.session_state["selected_factor"]

    if st.button("← Back to Dashboard", key="back_btn"):
        st.session_state["page"] = "dashboard"
        st.session_state["selected_asset"] = None
        st.session_state["selected_factor"] = None
        st.rerun()

    if not asset or not factor or asset not in all_data:
        st.warning("Invalid selection. Please go back to the dashboard.")
        return

    data = all_data[asset]
    asset_score: AssetRiskScore = data["score"]
    factor_scores: dict[str, float] = data["factor_scores"]
    factor_articles: list[dict] = data["factor_articles"].get(factor, [])

    render_factor_report(asset_score, factor, factor_scores, factor_articles)


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def main() -> None:
    with st.spinner("Loading asset data…"):
        all_data = _load_all_data()

    if st.session_state["page"] == "factor_report":
        _render_factor_report_page(all_data)
    else:
        _render_dashboard(all_data)


if __name__ == "__main__":
    main()
