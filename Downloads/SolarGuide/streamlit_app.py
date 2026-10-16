# -*- coding: utf-8 -*-
"""SolarGuide AI - Solar Monitoring & Predictive Maintenance dashboard.

Run locally:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run streamlit_app_cm.py

Keep these two files in the same folder as this script:
    Plant_2_Generation_Data.csv
    Plant_2_Weather_Sensor_Data.csv

The data pipeline and models are unchanged from the original notebook
(Random Forest -> loss -> Isolation Forest -> health score -> failure risk).
Only the interface was redesigned, using the CMU Africa / Tartan palette.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarGuide AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS (from the CMU Africa / Tartan PowerPoint template)
# ----------------------------------------------------------------------------
NAVY = "#002C49"       # template background navy
NAVY_DEEP = "#001A2C"
RED = "#C41230"        # tartan red
GREEN = "#2FA66A"      # tartan green
AMBER = "#F2A93B"      # warning
STEEL = "#5B7C93"
MIST = "#EEF2F6"       # page background
INK = "#0B2233"
MUTED = "#5E6E7C"

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

DATA_DIR = Path(__file__).resolve().parent

st.markdown(
    f"""
<style>
html, body, [class*="css"] {{ font-family: {FONT}; }}
.stApp {{ background: {MIST}; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DEEP} 100%);
    border-right: 4px solid {RED};
}}
[data-testid="stSidebar"] * {{ color: #E8EEF3; }}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{ gap: 6px; }}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 14px;
    width: 100%;
    transition: all .15s ease;
}}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {{
    background: rgba(255,255,255,0.12);
    transform: translateX(3px);
}}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {{ display: none; }}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {{
    background: {RED};
    border-color: {RED};
    box-shadow: 0 6px 18px rgba(196,18,48,0.45);
}}
.brand {{ display:flex; align-items:center; gap:12px; margin: 4px 0 18px 0; }}
.brand .logo {{
    width:44px; height:44px; border-radius:12px; font-size:26px;
    background: linear-gradient(135deg, {AMBER}, {RED});
    display:flex; align-items:center; justify-content:center;
}}
.brand .name {{ font-size:22px; font-weight:800; letter-spacing:.3px; color:#fff; line-height:1.1; }}
.brand .tag {{ font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#8FB0C6; }}
.side-note {{
    margin-top: 22px; padding: 14px; border-radius: 12px; font-size: 12px; line-height: 1.5;
    background: rgba(255,255,255,0.06); border-left: 3px solid {GREEN}; color:#BFD0DC;
}}

/* ---------- Hero ---------- */
.hero {{
    position: relative; overflow: hidden; border-radius: 20px; padding: 30px 36px; margin-bottom: 22px;
    background: linear-gradient(110deg, {NAVY} 0%, #0A3F63 100%);
    color: #fff; box-shadow: 0 10px 30px rgba(0,44,73,0.25);
}}
.hero::after {{
    content:""; position:absolute; inset:0; pointer-events:none; opacity:.55;
    background:
      repeating-linear-gradient(115deg, transparent 0 22px, rgba(196,18,48,.28) 22px 24px, transparent 24px 46px),
      repeating-linear-gradient(65deg, transparent 0 30px, rgba(47,166,106,.22) 30px 32px, transparent 32px 60px);
    -webkit-mask-image: linear-gradient(90deg, transparent 35%, #000 100%);
            mask-image: linear-gradient(90deg, transparent 35%, #000 100%);
}}
.hero h1 {{ margin:0; font-size:34px; font-weight:800; color:#fff; position:relative; z-index:1; }}
.hero p  {{ margin:6px 0 0 0; font-size:16px; color:#BFD6E6; position:relative; z-index:1; max-width:760px; }}
.hero .pill {{
    display:inline-block; position:relative; z-index:1; margin-bottom:10px;
    padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; background:rgba(255,255,255,.14); color:#fff;
}}

/* ---------- KPI cards ---------- */
.kpi {{
    background:#fff; border-radius:16px; padding:18px 20px 16px 20px; height:100%;
    box-shadow: 0 2px 10px rgba(0,44,73,0.07); border-top: 5px solid var(--c);
    transition: transform .15s ease, box-shadow .15s ease;
}}
.kpi:hover {{ transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,44,73,0.15); }}
.kpi .l {{ font-size:12px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:{MUTED}; }}
.kpi .v {{ font-size:34px; font-weight:800; color:{INK}; margin-top:6px; line-height:1.1; }}
.kpi .s {{ font-size:13px; color:var(--c); margin-top:4px; font-weight:600; }}

/* ---------- Panels ---------- */
.panel {{
    background:#fff; border-radius:16px; padding:22px 26px; margin-bottom:6px;
    box-shadow: 0 2px 10px rgba(0,44,73,0.07);
}}
.panel h3 {{ margin:0 0 8px 0; font-size:18px; color:{NAVY}; }}
.panel p, .panel li {{ color:#2B3F4F; font-size:15px; line-height:1.65; }}
.sec {{
    display:flex; align-items:center; gap:10px; margin: 26px 0 12px 0;
    font-size:20px; font-weight:800; color:{NAVY};
}}
.sec::before {{ content:""; width:6px; height:24px; border-radius:3px; background:{RED}; }}

/* ---------- Badges & spotlight ---------- */
.badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:800; color:#fff; }}
.b-low {{ background:{GREEN}; }} .b-med {{ background:{AMBER}; color:#3a2a00; }} .b-high {{ background:{RED}; }}
.spot {{
    border-radius:18px; padding:26px 30px; color:#fff;
    background: linear-gradient(120deg, {NAVY} 0%, #0A3F63 100%);
    box-shadow: 0 10px 28px rgba(0,44,73,.25); border-left: 8px solid {RED};
}}
.spot .t {{ font-size:12px; letter-spacing:2px; text-transform:uppercase; color:#8FB0C6; font-weight:700; }}
.spot .n {{ font-size:30px; font-weight:800; margin:2px 0 14px 0; }}
.spot .g {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; }}
.spot .c {{ background:rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; }}
.spot .c .k {{ font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#9DBBCF; }}
.spot .c .x {{ font-size:22px; font-weight:800; margin-top:2px; }}
.scale span {{ margin-right:18px; font-weight:600; }}

/* ---------- Native widgets ---------- */
[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; box-shadow:0 2px 10px rgba(0,44,73,0.07); }}
.stDownloadButton button, .stButton button {{
    background:{NAVY}; color:#fff; border:none; border-radius:10px; padding:8px 18px; font-weight:700;
}}
.stDownloadButton button:hover, .stButton button:hover {{ background:{RED}; color:#fff; }}
@media (max-width: 900px) {{ .spot .g {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------


def hero(title: str, subtitle: str, pill: str = "SolarGuide AI"):
    st.markdown(
        f'<div class="hero"><span class="pill">{pill}</span><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value, sub: str = "", color: str = NAVY):
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}&nbsp;</div></div>',
        unsafe_allow_html=True,
    )


def section(text: str):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def badge(level: str) -> str:
    cls = {"Low": "b-low", "Medium": "b-med", "High": "b-high"}.get(level, "b-med")
    return f'<span class="badge {cls}">{level}</span>'


def style_fig(fig, height: int = 420, title: str = None):
    """Apply the template look to every Plotly chart."""
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=18, color=NAVY)) if title else None,
        font=dict(family=FONT, color=INK),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=24, r=24, t=60 if title else 24, b=24),
        colorway=[NAVY, RED, GREEN, AMBER, STEEL],
        legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor=NAVY, font_color="#fff"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D5DEE6")
    fig.update_yaxes(gridcolor="#E6ECF1", zeroline=False)
    return fig


def chart(fig, height: int = 420, title: str = None):
    st.plotly_chart(style_fig(fig, height, title), use_container_width=True)


RISK_SCALE_GOOD_HIGH = [[0, RED], [0.5, AMBER], [1, GREEN]]
RISK_SCALE_BAD_HIGH = [[0, GREEN], [0.5, AMBER], [1, RED]]

# ----------------------------------------------------------------------------
# DATA + AI PIPELINE (logic unchanged; now cached so page changes are instant)
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data():
    """Read generation/weather CSV files, parse timestamps, and join matching records."""
    gen_path = DATA_DIR / "Plant_2_Generation_Data.csv"
    wx_path = DATA_DIR / "Plant_2_Weather_Sensor_Data.csv"

    generation = pd.read_csv(gen_path)
    weather = pd.read_csv(wx_path)

    generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"])
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

    # Keep only records whose timestamp and plant ID match in both datasets.
    merged = pd.merge(generation, weather, on=["DATE_TIME", "PLANT_ID"], how="inner")

    # Friendly inverter names: INV-1, INV-2, ... INV-N (stable: based on sorted source keys).
    # The original serial key is kept in ORIGINAL_SOURCE_KEY for traceability.
    merged["ORIGINAL_SOURCE_KEY"] = merged["SOURCE_KEY_x"]
    keys = sorted(merged["SOURCE_KEY_x"].unique())
    name_map = {key: f"INV-{i}" for i, key in enumerate(keys, start=1)}
    merged["SOURCE_KEY_x"] = merged["SOURCE_KEY_x"].map(name_map)
    return merged


def failure_level(prob):
    if prob < 30:
        return "Low"
    elif prob < 60:
        return "Medium"
    return "High"


def recommend_action(prob):
    if prob >= 80:
        return "Immediate Inspection"
    elif prob >= 60:
        return "Inspect Within 24 Hours"
    elif prob >= 40:
        return "Schedule Maintenance"
    return "Monitor"


def status(loss):
    if loss < 10:
        return "Normal"
    elif loss < 20:
        return "Warning"
    return "Critical"


def health_status(score):
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 75:
        return "🟡 Good"
    elif score >= 50:
        return "🟠 Warning"
    return "🔴 Critical"


@st.cache_data(show_spinner=False)
def run_pipeline(data: pd.DataFrame):
    features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    target = "AC_POWER"

    clean = data.dropna(subset=features + [target]).copy()

    # ---- Random Forest: expected power ----
    X, y = clean[features], clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    clean["EXPECTED_POWER"] = model.predict(clean[features])

    # Hold-out accuracy (added so the model can be judged, not just trusted).
    pred_test = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, pred_test)),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rows": int(len(clean)),
    }
    importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})

    # ---- Performance loss ----
    clean["LOSS_PERCENT"] = np.where(
        clean["EXPECTED_POWER"] > 0,
        ((clean["EXPECTED_POWER"] - clean["AC_POWER"]) / clean["EXPECTED_POWER"]) * 100,
        0,
    )

    # ---- Isolation Forest: anomalies (1 normal, -1 anomaly) ----
    anomaly_features = ["AC_POWER", "DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE"]
    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(clean[anomaly_features])
    clean["ANOMALY"] = iso.predict(clean[anomaly_features])

    # ---- Health score ----
    clean["RISK_SCORE"] = clean["LOSS_PERCENT"].clip(lower=0)
    clean.loc[clean["ANOMALY"] == -1, "RISK_SCORE"] += 50
    clean["RISK_SCORE"] = clean["RISK_SCORE"].clip(upper=100)
    clean["HEALTH_SCORE"] = 100 - clean["RISK_SCORE"]

    # ---- Failure probability (rule-based risk indicator, not a calibrated probability) ----
    clean["FAILURE_PROBABILITY"] = 100 - clean["HEALTH_SCORE"]
    clean.loc[clean["MODULE_TEMPERATURE"] > 55, "FAILURE_PROBABILITY"] += 10
    clean.loc[clean["ANOMALY"] == -1, "FAILURE_PROBABILITY"] += 15
    clean["FAILURE_PROBABILITY"] = clean["FAILURE_PROBABILITY"].clip(upper=100)

    clean["FAILURE_LEVEL"] = clean["FAILURE_PROBABILITY"].apply(failure_level)
    clean["ACTION"] = clean["FAILURE_PROBABILITY"].apply(recommend_action)
    clean["STATUS"] = clean["LOSS_PERCENT"].apply(status)

    # ---- Rule-based diagnosis (same rule order as the original, vectorised) ----
    overheating = (clean["LOSS_PERCENT"] > 20) & (clean["MODULE_TEMPERATURE"] > 55)
    dust = (clean["LOSS_PERCENT"] > 20) & (clean["IRRADIATION"] > 0.5)
    inverter_fault = clean["DC_POWER"] > clean["AC_POWER"] * 1.2
    clean["RECOMMENDATION"] = np.select(
        [overheating, dust, inverter_fault],
        ["Possible overheating", "Possible dust or shading", "Possible inverter fault"],
        default="Normal",
    )
    return clean, importance, metrics


# ----------------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------------
try:
    with st.spinner("Loading plant data..."):
        data = load_data()
except FileNotFoundError as err:
    st.error(
        f"Data file not found: **{Path(str(err.filename)).name}**. Place "
        "`Plant_2_Generation_Data.csv` and `Plant_2_Weather_Sensor_Data.csv` "
        "in the same folder as this script, then refresh."
    )
    st.stop()

with st.spinner("Training Random Forest and Isolation Forest models (first run only)..."):
    clean, importance, metrics = run_pipeline(data)

ID = "SOURCE_KEY_x"

# Inverter-level summary used across pages.
inv = (
    clean.groupby(ID)
    .agg(
        AC_POWER=("AC_POWER", "mean"),
        EXPECTED_POWER=("EXPECTED_POWER", "mean"),
        HEALTH_SCORE=("HEALTH_SCORE", "mean"),
        FAILURE_PROBABILITY=("FAILURE_PROBABILITY", "mean"),
        ANOMALIES=("ANOMALY", lambda s: int((s == -1).sum())),
    )
    .reset_index()
)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
NAV = {
    "📊  Plant Overview": "Plant Overview",
    "🔌  Inverter Health": "Inverter Health",
    "🤖  AI Predictions": "AI Predictions",
    "🚨  Maintenance Alerts": "Maintenance Alerts",
    "❤️  Asset Health": "Asset Health",
    "🔮  Failure Forecast": "Failure Forecast",
}
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">☀️</div><div>'
        '<div class="name">SolarGuide AI</div>'
        '<div class="tag">Predictive Maintenance</div></div></div>',
        unsafe_allow_html=True,
    )
    choice = st.radio("Navigate", list(NAV.keys()), label_visibility="collapsed")
    page = NAV[choice]
    st.markdown(
        f'<div class="side-note"><b>Live model status</b><br>'
        f'Random Forest R²: <b>{metrics["r2"]:.3f}</b><br>'
        f'Records analysed: <b>{metrics["rows"]:,}</b><br>'
        f'Inverters: <b>{len(inv)}</b></div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# PAGE 1: PLANT OVERVIEW
# ----------------------------------------------------------------------------
if page == "Plant Overview":
    hero(
        "Plant Overview",
        "Real-time health, power performance and AI risk signals across the entire solar plant.",
        "Executive dashboard",
    )

    total_power = clean["AC_POWER"].sum()
    avg_power = clean["AC_POWER"].mean()
    avg_temp = clean["MODULE_TEMPERATURE"].mean()
    plant_health = clean["HEALTH_SCORE"].mean()
    high_risk_assets = int((clean["FAILURE_PROBABILITY"] >= 60).sum())
    anomaly_count = int((clean["ANOMALY"] == -1).sum())
    asset_count = clean[ID].nunique()
    avg_failure_risk = clean["FAILURE_PROBABILITY"].mean()

    h_color = GREEN if plant_health >= 90 else AMBER if plant_health >= 75 else RED
    c = st.columns(5)
    kpi(c[0], "Plant Health", f"{plant_health:.1f}%", "Average health score", h_color)
    kpi(c[1], "Assets Monitored", asset_count, "Inverters online", NAVY)
    kpi(c[2], "Anomalies Detected", f"{anomaly_count:,}", "Isolation Forest", AMBER)
    kpi(c[3], "High Risk Assets", f"{high_risk_assets:,}", "Failure risk ≥ 60%", RED)
    kpi(c[4], "Avg Failure Risk", f"{avg_failure_risk:.1f}%", "Across all records", STEEL)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c = st.columns(3)
    kpi(c[0], "⚡ Total AC Power", f"{total_power:,.0f}", "Cumulative output", GREEN)
    kpi(c[1], "📈 Average Power", f"{avg_power:.2f}", "Per record", NAVY)
    kpi(c[2], "🌡 Avg Module Temp", f"{avg_temp:.2f}°C", "Panel surface", AMBER)

    section("Executive Summary")
    st.markdown(
        f"""<div class="panel"><p>
        The solar plant is currently operating with an average health score of
        <b>{plant_health:.1f}%</b>. <b>{high_risk_assets:,}</b> records have been identified as
        high-risk and may require maintenance attention. A total of <b>{anomaly_count:,}</b>
        anomalous operating events have been detected by the AI monitoring system.
        </p></div>""",
        unsafe_allow_html=True,
    )

    section("Plant Power Output")
    power_time = (
        clean.groupby("DATE_TIME")[["AC_POWER", "EXPECTED_POWER"]].sum().reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=power_time["DATE_TIME"], y=power_time["EXPECTED_POWER"], name="Expected (AI)",
            line=dict(color=RED, width=1.6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=power_time["DATE_TIME"], y=power_time["AC_POWER"], name="Actual",
            line=dict(color=NAVY, width=2), fill="tozeroy", fillcolor="rgba(0,44,73,0.10)",
        )
    )
    fig.update_yaxes(title="AC power (kW)")
    chart(fig, 420)

    left, right = st.columns([1, 1])
    with left:
        section("Plant Health Gauge")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=plant_health,
                number=dict(suffix="%", font=dict(size=54, color=NAVY)),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=NAVY, thickness=0.28),
                    steps=[
                        dict(range=[0, 75], color="#F4C7CE"),
                        dict(range=[75, 90], color="#FBE6C2"),
                        dict(range=[90, 100], color="#C9ECDB"),
                    ],
                ),
            )
        )
        chart(gauge, 330)
    with right:
        section("Risk Distribution")
        lvl = clean["FAILURE_LEVEL"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        lvl.columns = ["Level", "Records"]
        fig = px.pie(
            lvl, names="Level", values="Records", hole=0.62, color="Level",
            color_discrete_map={"Low": GREEN, "Medium": AMBER, "High": RED},
        )
        fig.update_traces(textinfo="percent+label", sort=False)
        chart(fig, 330)

    section("Top 5 Inverters Needing Attention")
    top5 = inv.sort_values("FAILURE_PROBABILITY", ascending=False).head(5)
    cols = st.columns(5)
    for col, (_, r) in zip(cols, top5.iterrows()):
        lvl_name = failure_level(r["FAILURE_PROBABILITY"])
        col.markdown(
            f'<div class="kpi" style="--c:{RED if lvl_name=="High" else AMBER if lvl_name=="Medium" else GREEN}">'
            f'<div class="l">{r[ID]}</div><div class="v">{r["FAILURE_PROBABILITY"]:.0f}%</div>'
            f'<div class="s">{badge(lvl_name)}</div></div>',
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# PAGE 2: INVERTER HEALTH
# ----------------------------------------------------------------------------
elif page == "Inverter Health":
    hero(
        "Inverter Health",
        "Compare every inverter and drill into any unit to see actual vs AI-expected output.",
        "Inverter analytics",
    )
    inverter = inv[[ID, "AC_POWER"]]
    best = inverter.loc[inverter["AC_POWER"].idxmax()]
    worst = inverter.loc[inverter["AC_POWER"].idxmin()]

    c = st.columns(3)
    kpi(c[0], "🏆 Best Inverter", best[ID], f"{best['AC_POWER']:.1f} avg power", GREEN)
    kpi(c[1], "⚠️ Weakest Inverter", worst[ID], f"{worst['AC_POWER']:.1f} avg power", RED)
    kpi(c[2], "Spread", f"{best['AC_POWER'] - worst['AC_POWER']:.1f}", "Best minus weakest", AMBER)

    section("Average AC Power by Inverter")
    fig = px.bar(
        inverter.sort_values("AC_POWER", ascending=False), x=ID, y="AC_POWER",
        color="AC_POWER", color_continuous_scale=RISK_SCALE_GOOD_HIGH,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Avg AC power")
    fig.update_xaxes(showticklabels=False)
    chart(fig, 380)

    section("Inverter Drill-down")
    pick = st.selectbox("Choose an inverter", inv.sort_values("FAILURE_PROBABILITY", ascending=False)[ID])
    sub = clean[clean[ID] == pick].sort_values("DATE_TIME")
    row = inv[inv[ID] == pick].iloc[0]
    c = st.columns(4)
    kpi(c[0], "Health Score", f"{row['HEALTH_SCORE']:.1f}%", "", GREEN if row["HEALTH_SCORE"] >= 75 else RED)
    kpi(c[1], "Failure Risk", f"{row['FAILURE_PROBABILITY']:.1f}%", "", RED if row["FAILURE_PROBABILITY"] >= 60 else AMBER)
    kpi(c[2], "Anomalies", int(row["ANOMALIES"]), "Flagged events", AMBER)
    kpi(c[3], "Suggested Action", "", recommend_action(row["FAILURE_PROBABILITY"]), NAVY)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["DATE_TIME"], y=sub["EXPECTED_POWER"], name="Expected", line=dict(color=RED, dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=sub["DATE_TIME"], y=sub["AC_POWER"], name="Actual", line=dict(color=NAVY, width=2)))
    an = sub[sub["ANOMALY"] == -1]
    fig.add_trace(go.Scatter(x=an["DATE_TIME"], y=an["AC_POWER"], name="Anomaly", mode="markers", marker=dict(color=AMBER, size=8, line=dict(color=NAVY, width=1))))
    fig.update_yaxes(title="AC power")
    chart(fig, 400)

# ----------------------------------------------------------------------------
# PAGE 3: AI PREDICTIONS
# ----------------------------------------------------------------------------
elif page == "AI Predictions":
    hero(
        "AI Power Prediction",
        "A Random Forest learns what each inverter should produce from sunlight and temperature.",
        "Machine learning",
    )
    c = st.columns(3)
    kpi(c[0], "Model R²", f"{metrics['r2']:.3f}", "Hold-out test set", GREEN)
    kpi(c[1], "Mean Abs. Error", f"{metrics['mae']:.1f}", "AC power units", AMBER)
    kpi(c[2], "Training Records", f"{metrics['rows']:,}", "Cleaned rows", NAVY)

    left, right = st.columns([3, 2])
    with left:
        section("Expected vs Actual Power")
        sample = clean.sample(min(2000, len(clean)), random_state=42)
        fig = px.scatter(sample, x="EXPECTED_POWER", y="AC_POWER", opacity=0.55, color_discrete_sequence=[NAVY])
        m = float(max(sample["EXPECTED_POWER"].max(), sample["AC_POWER"].max()))
        fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="Perfect prediction", line=dict(color=RED, dash="dash")))
        fig.update_xaxes(title="Expected power (AI)")
        fig.update_yaxes(title="Actual power")
        chart(fig, 460)
    with right:
        section("Feature Importance")
        imp = importance.sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=[[0, STEEL], [1, NAVY]])
        fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
        chart(fig, 460)

# ----------------------------------------------------------------------------
# PAGE 4: MAINTENANCE ALERTS
# ----------------------------------------------------------------------------
elif page == "Maintenance Alerts":
    hero(
        "Maintenance Alerts",
        "Every warning, critical loss and anomaly - with the AI's diagnosis and recommended action.",
        "Predictive maintenance",
    )
    alerts = clean[(clean["STATUS"] != "Normal") | (clean["ANOMALY"] == -1)]

    health_score = 100 - clean["LOSS_PERCENT"].clip(lower=0).mean()
    c = st.columns(4)
    kpi(c[0], "Plant Health Score", f"{health_score:.1f}%", "Loss-based", GREEN if health_score >= 90 else AMBER if health_score >= 75 else RED)
    kpi(c[1], "Total Alerts", f"{len(alerts):,}", "Warning, critical or anomaly", NAVY)
    kpi(c[2], "Critical", f"{int((alerts['STATUS'] == 'Critical').sum()):,}", "Loss ≥ 20%", RED)
    kpi(c[3], "Anomalies", f"{int((alerts['ANOMALY'] == -1).sum()):,}", "Isolation Forest", AMBER)

    st.markdown(
        '<div class="panel scale" style="margin-top:16px"><b>Plant Health Scale:</b>&nbsp;&nbsp;'
        '<span>🟢 90-100 Excellent</span><span>🟡 75-89 Good</span><span>🔴 &lt;75 Critical</span></div>',
        unsafe_allow_html=True,
    )

    section("Alert Feed")
    f1, f2 = st.columns(2)
    action_filter = f1.multiselect("Action", sorted(alerts["ACTION"].unique()), default=sorted(alerts["ACTION"].unique()))
    diag_filter = f2.multiselect("Diagnosis", sorted(alerts["RECOMMENDATION"].unique()), default=sorted(alerts["RECOMMENDATION"].unique()))
    view = alerts[alerts["ACTION"].isin(action_filter) & alerts["RECOMMENDATION"].isin(diag_filter)]
    view = view.sort_values("FAILURE_PROBABILITY", ascending=False)

    cols = ["DATE_TIME", ID, "AC_POWER", "EXPECTED_POWER", "LOSS_PERCENT", "RECOMMENDATION", "FAILURE_PROBABILITY", "ACTION"]
    st.caption(f"Showing the {min(len(view), 1500):,} highest-risk of {len(view):,} matching alerts.")
    st.dataframe(
        view[cols].head(1500),
        use_container_width=True,
        height=460,
        hide_index=True,
        column_config={
            "FAILURE_PROBABILITY": st.column_config.ProgressColumn("Failure risk", min_value=0, max_value=100, format="%.0f%%"),
            "LOSS_PERCENT": st.column_config.NumberColumn("Loss %", format="%.1f"),
            "AC_POWER": st.column_config.NumberColumn("Actual", format="%.1f"),
            "EXPECTED_POWER": st.column_config.NumberColumn("Expected", format="%.1f"),
            ID: "Inverter",
        },
    )
    st.download_button("⬇️ Download alerts (CSV)", view[cols].to_csv(index=False).encode(), "maintenance_alerts.csv", "text/csv")

# ----------------------------------------------------------------------------
# PAGE 5: ASSET HEALTH
# ----------------------------------------------------------------------------
elif page == "Asset Health":
    hero(
        "Asset Health Dashboard",
        "Rank every inverter by health score and surface the assets that need action first.",
        "Asset management",
    )
    leaderboard = inv[[ID, "HEALTH_SCORE"]].sort_values("HEALTH_SCORE", ascending=False)
    critical = leaderboard[leaderboard["HEALTH_SCORE"] < 60]

    asset_summary = inv[[ID, "HEALTH_SCORE", "FAILURE_PROBABILITY"]].copy()
    asset_summary["STATUS"] = asset_summary["HEALTH_SCORE"].apply(health_status)
    asset_summary.columns = ["Inverter", "Health Score (%)", "Failure Probability (%)", "Status"]
    asset_summary = asset_summary.sort_values("Health Score (%)", ascending=False)
    critical_assets = asset_summary[asset_summary["Health Score (%)"] < 50]

    c = st.columns(4)
    kpi(c[0], "Healthiest", f"{leaderboard['HEALTH_SCORE'].max():.1f}%", "Top inverter", GREEN)
    kpi(c[1], "Lowest", f"{leaderboard['HEALTH_SCORE'].min():.1f}%", "Weakest inverter", RED)
    kpi(c[2], "Below 60%", len(critical), "Critical assets", RED if len(critical) else GREEN)
    kpi(c[3], "Below 50%", len(critical_assets), "Immediate attention", RED if len(critical_assets) else GREEN)

    section("Inverter Health Ranking")
    fig = px.bar(leaderboard, x=ID, y="HEALTH_SCORE", color="HEALTH_SCORE", color_continuous_scale=RISK_SCALE_GOOD_HIGH, range_color=[0, 100])
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Health score")
    fig.update_xaxes(showticklabels=False)
    chart(fig, 400)

    section("Critical Assets (health below 60%)")
    if len(critical):
        st.dataframe(critical, use_container_width=True, hide_index=True)
    else:
        st.success("No inverter is below the 60% critical threshold. 🎉")

    section("Asset Summary")
    st.dataframe(
        asset_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Health Score (%)": st.column_config.ProgressColumn("Health Score (%)", min_value=0, max_value=100, format="%.1f"),
            "Failure Probability (%)": st.column_config.ProgressColumn("Failure Probability (%)", min_value=0, max_value=100, format="%.1f"),
        },
    )

    section("🔴 Assets Requiring Immediate Attention")
    if len(critical_assets):
        st.dataframe(critical_assets, use_container_width=True, hide_index=True)
    else:
        st.success("No inverter is below the 50% immediate-attention threshold.")

# ----------------------------------------------------------------------------
# PAGE 6: FAILURE FORECAST
# ----------------------------------------------------------------------------
elif page == "Failure Forecast":
    hero(
        "Failure Risk Forecast",
        "Which inverters are most likely to fail next - and what to do about it.",
        "Predictive insight",
    )
    failure_ranking = (
        inv[[ID, "FAILURE_PROBABILITY"]].sort_values("FAILURE_PROBABILITY", ascending=False).reset_index(drop=True)
    )

    # Spotlight: highest-risk inverter (same fields as the pitch-deck example).
    top = inv.sort_values("FAILURE_PROBABILITY", ascending=False).iloc[0]
    loss_pct = (top["EXPECTED_POWER"] - top["AC_POWER"]) / top["EXPECTED_POWER"] * 100 if top["EXPECTED_POWER"] > 0 else 0
    lvl = failure_level(top["FAILURE_PROBABILITY"])
    st.markdown(
        f"""<div class="spot"><div class="t">AI insight spotlight · highest risk inverter</div>
        <div class="n">{top[ID]} &nbsp;{badge(lvl)}</div>
        <div class="g">
          <div class="c"><div class="k">Expected power</div><div class="x">{top['EXPECTED_POWER']:.0f}</div></div>
          <div class="c"><div class="k">Actual power</div><div class="x">{top['AC_POWER']:.0f}</div></div>
          <div class="c"><div class="k">Performance loss</div><div class="x">{loss_pct:.1f}%</div></div>
          <div class="c"><div class="k">Anomalies</div><div class="x">{int(top['ANOMALIES']):,}</div></div>
          <div class="c"><div class="k">Health score</div><div class="x">{top['HEALTH_SCORE']:.0f}%</div></div>
          <div class="c"><div class="k">Failure probability</div><div class="x">{top['FAILURE_PROBABILITY']:.0f}%</div></div>
          <div class="c" style="grid-column: span 2"><div class="k">Suggested action</div><div class="x">{recommend_action(top['FAILURE_PROBABILITY'])}</div></div>
        </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c = st.columns(3)
    kpi(c[0], "Highest Failure Risk", f"{failure_ranking['FAILURE_PROBABILITY'].max():.1f}%", "Top inverter", RED)
    kpi(c[1], "High-risk Inverters", int((failure_ranking["FAILURE_PROBABILITY"] >= 60).sum()), "Risk ≥ 60%", RED)
    kpi(c[2], "Low-risk Inverters", int((failure_ranking["FAILURE_PROBABILITY"] < 30).sum()), "Risk < 30%", GREEN)

    section("Failure Probability Ranking")
    fig = px.bar(failure_ranking, x=ID, y="FAILURE_PROBABILITY", color="FAILURE_PROBABILITY", color_continuous_scale=RISK_SCALE_BAD_HIGH, range_color=[0, 100])
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Failure probability (%)")
    fig.update_xaxes(showticklabels=False)
    chart(fig, 400)

    def risk_color(prob):
        if prob >= 60:
            return "🔴 High"
        elif prob >= 30:
            return "🟡 Medium"
        return "🟢 Low"

    failure_ranking["Risk"] = failure_ranking["FAILURE_PROBABILITY"].apply(risk_color)
    section("Risk Table")
    st.dataframe(
        failure_ranking, use_container_width=True, hide_index=True,
        column_config={
            ID: "Inverter",
            "FAILURE_PROBABILITY": st.column_config.ProgressColumn("Failure probability", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    section("High Risk Records")
    high_risk_records = (
        clean[clean["FAILURE_PROBABILITY"] >= 60][[ID, "HEALTH_SCORE", "FAILURE_PROBABILITY", "ACTION"]]
        .sort_values("FAILURE_PROBABILITY", ascending=False)
    )
    st.caption(f"Showing the {min(len(high_risk_records), 1000):,} highest-risk of {len(high_risk_records):,} records.")
    st.dataframe(
        high_risk_records.head(1000), use_container_width=True, hide_index=True, height=380,
        column_config={
            ID: "Inverter",
            "HEALTH_SCORE": st.column_config.NumberColumn("Health", format="%.0f"),
            "FAILURE_PROBABILITY": st.column_config.ProgressColumn("Failure probability", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div style="margin-top:40px;padding:22px 26px;border-radius:16px;background:{NAVY};color:#BFD6E6;font-size:14px">
    <b style="color:#fff">SolarGuide AI</b> · Solar Monitoring &amp; Predictive Maintenance<br>
    Powered by Random Forest Regression · Isolation Forest Anomaly Detection · Health Score Engine · Failure Risk Forecasting
    </div>""",
    unsafe_allow_html=True,
)
