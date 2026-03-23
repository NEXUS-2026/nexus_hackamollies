"""
app.py
──────
Streamlit dashboard — AI-Powered Predictive Maintenance System.
Run with:  streamlit run frontend/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from backend.data_generator import generate_sensor_data, generate_component_data
from backend.models import (
    train_anomaly_detector, detect_anomalies,
    calculate_risk_score, calculate_risk_scores,
    train_failure_predictor, predict_failure,
    get_system_status,
)
from backend.graph_model import (
    build_system_graph, assign_risk_to_nodes,
    simulate_failure_propagation, DEFAULT_COMPONENTS,
    build_component_summary, COMPONENT_DETAILS,
    estimate_days_to_failure, estimate_maintenance_cost,
    get_system_health_score,
)
from frontend.visualizations import (
    plot_sensor_timeseries, plot_risk_heatmap,
    plot_network_graph, plot_risk_gauge,
    plot_aging_chart, plot_maintenance_timeline,
)


# ═══════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
    border: 1px solid rgba(99,102,241,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 { color:#e0e7ff; font-weight:700; margin:0 0 .25rem; font-size:1.85rem; }
.main-header p   { color:#a5b4fc; margin:0; font-size:1rem; }

.status-badge {
    display:inline-block; padding:.35rem 1rem; border-radius:20px;
    font-weight:600; font-size:.95rem; letter-spacing:.4px;
}

.metric-card {
    background: linear-gradient(145deg,#1e1e2f,#1a1d23);
    padding:1.1rem; border-radius:14px; text-align:center;
    border:1px solid #2a2d35; box-shadow:0 4px 16px rgba(0,0,0,.2);
    transition: transform .15s, border-color .2s;
}
.metric-card:hover { transform: translateY(-2px); border-color:rgba(99,102,241,.4); }
.metric-card h3 { color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:.25rem; }
.metric-card .value { font-size:1.8rem; font-weight:700; }
.metric-card .sublabel { color:#64748b; font-size:.74rem; margin-top:.1rem; }
.metric-card .trend { font-size:.78rem; margin-top:.15rem; }

.alert-box { padding:.85rem 1.2rem; border-radius:12px; margin-bottom:.7rem; font-weight:500; font-size:.9rem; }
.alert-warning  { background:rgba(255,145,0,.12); border-left:4px solid #ff9100; color:#c06000; }
.alert-critical { background:rgba(255,23,68,.12); border-left:4px solid #ff1744; color:#cc0022; }
.alert-ok       { background:rgba(0,200,83,.12); border-left:4px solid #00c853; color:#007a33; }
.alert-info     { background:rgba(99,102,241,.10); border-left:4px solid #6366f1; color:#4045bb; }

.section-header {
    color:#c7d2fe; font-size:1.18rem; font-weight:600;
    margin:1.4rem 0 .5rem; padding-bottom:.3rem;
    border-bottom:2px solid rgba(99,102,241,.3);
}
.explainer { color:#94a3b8; font-size:.88rem; margin-bottom:.7rem; line-height:1.45; }

.comp-card {
    background: linear-gradient(145deg,#1e1e2f,#1a1d23);
    padding:1rem 1.2rem; border-radius:14px;
    border:1px solid #2a2d35; box-shadow:0 4px 16px rgba(0,0,0,.2);
    margin-bottom:.6rem; transition: border-color .2s;
}
.comp-card:hover { border-color: rgba(99,102,241,.5); }
.comp-card h4 { color:#e0e7ff; margin:0 0 .45rem; font-size:1rem; }
.comp-card .detail-row {
    display:flex; justify-content:space-between;
    padding:.2rem 0; border-bottom:1px solid #1e2028; font-size:.84rem;
}
.comp-card .detail-row:last-child { border-bottom:none; }
.comp-card .detail-label { color:#64748b; }
.comp-card .detail-value { color:#c7d2fe; font-weight:500; }

/* Cost card */
.cost-card {
    background: linear-gradient(145deg,#1a2332,#162030);
    padding:1rem 1.2rem; border-radius:14px;
    border:1px solid rgba(99,102,241,.2);
    margin-bottom:.6rem;
}
.cost-card h4 { color:#a5b4fc; margin:0 0 .5rem; font-size:.95rem; }
.cost-card .cost-row {
    display:flex; justify-content:space-between; padding:.2rem 0;
    font-size:.84rem; border-bottom:1px solid #1a2535;
}
.cost-card .cost-row:last-child { border-bottom:none; }
.cost-label { color:#64748b; }
.cost-value { color:#e0e7ff; font-weight:600; }

/* Sidebar panel headers */
.panel-title {
    color:#c7d2fe; font-size:.95rem; font-weight:600;
    margin:.8rem 0 .4rem; display:flex; align-items:center; gap:.4rem;
}
.panel-divider {
    border:none; border-top:1px solid rgba(99,102,241,.15);
    margin:.8rem 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0e1117 0%,#151922 100%);
}

/* ══════════════════════════════════════════
   LIGHT MODE OVERRIDES
   Streamlit sets data-theme="light" on <html>
   ══════════════════════════════════════════ */
[data-theme="light"] section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#f0f4ff 0%,#e8eef8 100%) !important;
}
[data-theme="light"] .panel-title {
    color: #1e2a4a !important;
}
[data-theme="light"] .panel-divider {
    border-top-color: rgba(99,102,241,.2) !important;
}
[data-theme="light"] .section-header {
    color: #1e2a4a !important;
    border-bottom-color: rgba(99,102,241,.3) !important;
}
[data-theme="light"] .explainer {
    color: #3d5068 !important;
}
[data-theme="light"] .metric-card {
    background: linear-gradient(145deg,#ffffff,#f3f6fb) !important;
    border-color: #d8e0f0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.08) !important;
}
[data-theme="light"] .metric-card h3 {
    color: #5a7090 !important;
}
[data-theme="light"] .comp-card {
    background: linear-gradient(145deg,#ffffff,#f3f6fb) !important;
    border-color: #d8e0f0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.08) !important;
}
[data-theme="light"] .comp-card h4 {
    color: #1a2440 !important;
}
[data-theme="light"] .comp-card .detail-row {
    border-bottom-color: #e4eaf5 !important;
}
[data-theme="light"] .comp-card .detail-label {
    color: #5a7090 !important;
}
[data-theme="light"] .comp-card .detail-value {
    color: #1a2440 !important;
}
[data-theme="light"] .cost-card {
    background: linear-gradient(145deg,#edf2ff,#e4ecfb) !important;
    border-color: rgba(99,102,241,.25) !important;
}
[data-theme="light"] .cost-card h4 {
    color: #3a42a0 !important;
}
[data-theme="light"] .cost-card .cost-row {
    border-bottom-color: #d0d8f0 !important;
}
[data-theme="light"] .cost-label {
    color: #5a6a90 !important;
}
[data-theme="light"] .cost-value {
    color: #1a2440 !important;
}
[data-theme="light"] .alert-warning  { color: #8a4a00 !important; }
[data-theme="light"] .alert-critical  { color: #aa0018 !important; }
[data-theme="light"] .alert-ok        { color: #006030 !important; }
[data-theme="light"] .alert-info      { color: #2a30a0 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — Enhanced Control Panel
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── LIVE RISK MONITOR HERO BUTTON ───────
    st.markdown(
        """
        <a href="http://localhost:8080/component_risk_monitor.html"
           style="display:block; padding:16px 14px; margin-bottom:18px;
                  background:linear-gradient(135deg,rgba(0,212,255,0.15) 0%,rgba(0,180,220,0.08) 100%);
                  border:1.5px solid rgba(0,212,255,0.55);
                  border-radius:14px; text-decoration:none;
                  box-shadow:0 0 24px rgba(0,212,255,0.2), inset 0 0 20px rgba(0,212,255,0.04);
                  transition:all .2s;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:1.5rem;line-height:1;">⚡</span>
            <span style="font-size:1.05rem;font-weight:800;color:#00d4ff;
                         letter-spacing:.08em;text-transform:uppercase;">Live Risk Monitor</span>
            <span style="margin-left:auto;background:rgba(255,34,68,0.18);
                         border:1px solid rgba(255,34,68,0.45);border-radius:20px;
                         padding:2px 9px;font-size:.7rem;font-weight:700;
                         color:#ff4466;letter-spacing:.1em;
                         animation:pulse-sidebar 1.4s ease-in-out infinite;">LIVE</span>
          </div>
          <div style="font-size:.74rem;color:rgba(0,212,255,0.6);padding-left:36px;
                       letter-spacing:.03em;">
            Real-time sensor gauges &amp; anomaly log
          </div>
        </a>
        <style>
          @keyframes pulse-sidebar {
            0%,100%{opacity:1} 50%{opacity:0.45}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🎛️ Control Panel")

    # ── 1. DATA SIMULATION CONTROLS ─────────
    st.markdown('<p class="panel-title">📊 Data Simulation</p>',
                unsafe_allow_html=True)

    n_points = st.slider(
        "Sensor readings", 100, 1000, 500, step=50,
        help="Historical data window size.",
    )
    anomaly_ratio = st.slider(
        "Anomaly rate", 0.01, 0.25, 0.05, step=0.01,
        help="Higher = more simulated faults.",
    )
    time_window = st.selectbox(
        "📅 Time window",
        ["All Data", "Last 1 hour", "Last 4 hours", "Last 8 hours"],
        help="Filter the sensor charts to a recent time window.",
    )

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 2. COMPONENT INSPECTOR ──────────────
    st.markdown('<p class="panel-title">🔍 Component Inspector</p>',
                unsafe_allow_html=True)

    selected_component = st.selectbox(
        "Select component",
        ["🌐 System Overview"] + DEFAULT_COMPONENTS,
        help="Deep-dive into a specific component.",
    )

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 3. COMPONENT COMPARISON ─────────────
    st.markdown('<p class="panel-title">⚖️ Component Comparison</p>',
                unsafe_allow_html=True)

    compare_mode = st.toggle("Enable comparison mode", value=False)
    if compare_mode:
        comp_a = st.selectbox("Component A", DEFAULT_COMPONENTS, index=0,
                              key="comp_a")
        comp_b = st.selectbox("Component B", DEFAULT_COMPONENTS, index=1,
                              key="comp_b")
    else:
        comp_a = comp_b = None

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 4. FAILURE SIMULATION ───────────────
    st.markdown('<p class="panel-title">💥 Failure Simulation</p>',
                unsafe_allow_html=True)

    simulate_fail = st.toggle("Simulate failure", value=False)
    if simulate_fail:
        failed_component = st.selectbox(
            "Failed component", DEFAULT_COMPONENTS, key="fail_comp",
        )
        propagation_factor = st.slider(
            "Cascade strength", 0.3, 1.0, 0.7, step=0.05,
            help="1.0 = full pass-through, 0.3 = heavily dampened.",
        )
    else:
        failed_component = None
        propagation_factor = 0.7

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 5. ALERT THRESHOLDS ─────────────────
    st.markdown('<p class="panel-title">🎚️ Alert Sensitivity</p>',
                unsafe_allow_html=True)

    with st.expander("Customise thresholds", expanded=False):
        warn_threshold = st.slider(
            "⚠️ Warning threshold", 20, 60, 35,
            help="Risk score above this triggers a warning.",
        )
        crit_threshold = st.slider(
            "🚨 Critical threshold", 40, 90, 65,
            help="Risk score above this triggers a critical alert.",
        )
        maint_days_warn = st.slider(
            "🔧 Maintenance overdue (days)", 60, 365, 180,
            help="Alert when a component hasn't been serviced in this many days.",
        )

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 6. DISPLAY OPTIONS ──────────────────
    st.markdown('<p class="panel-title">🖥️ Display Options</p>',
                unsafe_allow_html=True)

    show_anomalies_only = st.checkbox("Show anomalies only", value=False)
    show_cost_analysis = st.checkbox("Show cost analysis", value=True)
    show_failure_timeline = st.checkbox("Show failure timeline", value=True)
    show_data_table = st.checkbox("Show data table", value=False)

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # ── 7. ACTIONS ──────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()
    with col_b:
        pass  # CSV download is in the main area

    st.markdown(
        f'<p style="color:#475569; font-size:.72rem; margin-top:.8rem; text-align:center;">'
        f'Last updated: {datetime.now().strftime("%H:%M:%S")}</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# DATA GENERATION & MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(n_points, anomaly_ratio):
    df = generate_sensor_data(n_points, anomaly_ratio)
    comp_data = generate_component_data(DEFAULT_COMPONENTS, n_points=200,
                                        anomaly_ratio=anomaly_ratio)
    iso_model, iso_scaler = train_anomaly_detector(df, contamination=anomaly_ratio)
    df["anomaly_label"] = detect_anomalies(iso_model, iso_scaler, df)
    df["risk_score"] = calculate_risk_scores(df)

    fail_clf, fail_scaler = train_failure_predictor(df)
    fail_df = predict_failure(fail_clf, fail_scaler, df)
    df["failure_prob"] = fail_df["failure_prob"]
    df["failure_pred"] = fail_df["failure_pred"]

    component_risks = {}
    comp_dfs = {}
    for comp, cdf in comp_data.items():
        iso_m, iso_s = train_anomaly_detector(cdf, contamination=anomaly_ratio)
        cdf["anomaly_label"] = detect_anomalies(iso_m, iso_s, cdf)
        cdf["risk_score"] = calculate_risk_scores(cdf)
        component_risks[comp] = round(cdf["risk_score"].mean(), 1)
        comp_dfs[comp] = cdf

    return df, component_risks, comp_dfs


with st.spinner("🔧 Generating sensor data & training models…"):
    df, component_risks, comp_dfs = load_data(n_points, anomaly_ratio)

comp_summary = build_component_summary()

# ── Apply time window filter ────────────────
if time_window != "All Data":
    hours_map = {"Last 1 hour": 1, "Last 4 hours": 4, "Last 8 hours": 8}
    hours = hours_map.get(time_window, 999)
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
    df_filtered = df[df["timestamp"] >= cutoff].copy()
else:
    df_filtered = df


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

latest_risk = df["risk_score"].iloc[-1]

# Use custom thresholds for status
if latest_risk >= crit_threshold:
    status_label, status_color = "🚨 Critical", "#ff1744"
elif latest_risk >= warn_threshold:
    status_label, status_color = "⚠️ At Risk", "#ff9100"
else:
    status_label, status_color = "✅ Normal", "#00c853"

system_health = get_system_health_score(component_risks)

st.markdown(f"""
<div class="main-header">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.8rem;">
        <div>
            <h1>🔧 AI Predictive Maintenance Dashboard</h1>
            <p>Monitor equipment health · Detect anomalies · Predict failures before they happen</p>
        </div>
        <div style="display:flex; gap:.6rem; align-items:center;">
            <span class="status-badge" style="background:{status_color}22; color:{status_color};
                  border:1px solid {status_color};">{status_label}</span>
            <span class="status-badge" style="background:rgba(99,102,241,.12); color:#a5b4fc;
                  border:1px solid rgba(99,102,241,.3);">🏥 Health: {system_health}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# KEY METRICS (6 cards)
# ═══════════════════════════════════════════════════════════════════════

anomaly_count  = int((df_filtered["anomaly_label"] == "Anomaly").sum())
avg_risk       = round(df_filtered["risk_score"].mean(), 1)
fail_pct       = round(df_filtered["failure_prob"].mean() * 100, 1)
total_readings = len(df_filtered)
healthy_comps  = sum(1 for s in component_risks.values() if s < warn_threshold)
total_comps    = len(component_risks)

# Risk trend: compare first-half avg vs second-half avg
half = len(df_filtered) // 2
if half > 0:
    first_half  = df_filtered["risk_score"].iloc[:half].mean()
    second_half = df_filtered["risk_score"].iloc[half:].mean()
    trend_diff  = second_half - first_half
    if trend_diff > 2:
        trend_icon, trend_color, trend_text = "📈", "#ff5252", f"+{trend_diff:.1f} rising"
    elif trend_diff < -2:
        trend_icon, trend_color, trend_text = "📉", "#00c853", f"{trend_diff:.1f} falling"
    else:
        trend_icon, trend_color, trend_text = "➡️", "#818cf8", "stable"
else:
    trend_icon, trend_color, trend_text = "➡️", "#818cf8", "stable"

# Total estimated cost
total_cost = sum(
    estimate_maintenance_cost(c, component_risks.get(c, 0))["cost"]
    for c in DEFAULT_COMPONENTS
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Current Risk</h3>
        <div class="value" style="color:{status_color};">{latest_risk:.0f}</div>
        <div class="sublabel">out of 100</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Risk Trend</h3>
        <div class="value" style="color:{trend_color};">{trend_icon}</div>
        <div class="sublabel">{trend_text}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Anomalies</h3>
        <div class="value" style="color:{'#ff1744' if anomaly_count > 10 else '#ff9100' if anomaly_count > 0 else '#00c853'};">{anomaly_count}</div>
        <div class="sublabel">{round(anomaly_count/max(1,total_readings)*100,1)}% of data</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Failure Prob</h3>
        <div class="value" style="color:{'#ff1744' if fail_pct > 30 else '#ff9100' if fail_pct > 10 else '#00c853'};">{fail_pct}%</div>
        <div class="sublabel">ML prediction</div>
    </div>""", unsafe_allow_html=True)

with c5:
    comp_color = "#00c853" if healthy_comps == total_comps else "#ff9100"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Components</h3>
        <div class="value" style="color:{comp_color};">{healthy_comps}/{total_comps}</div>
        <div class="sublabel">healthy</div>
    </div>""", unsafe_allow_html=True)

with c6:
    cost_color = "#00c853" if total_cost == 0 else "#ff9100" if total_cost < 50000 else "#ff5252"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Est. Maint Cost</h3>
        <div class="value" style="color:{cost_color};">${total_cost:,.0f}</div>
        <div class="sublabel">all components</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ALERTS (using custom thresholds)
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">📢 System Alerts</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="explainer">'
    'Real-time alerts based on your configured thresholds. '
    'Each alert is colour-coded by severity.'
    '</p>',
    unsafe_allow_html=True,
)

# ── 1. Overall system status ──
if latest_risk >= crit_threshold:
    st.markdown(
        '<div class="alert-box alert-critical">'
        '🚨 <strong>CRITICAL — Immediate Action Required</strong>'
        '<br><br>'
        'The latest sensor readings have exceeded the critical threshold '
        f'(risk score: <strong>{latest_risk:.0f}</strong> / '
        f'threshold: <strong>{crit_threshold}</strong>).'
        '<br><br>'
        'Shut down affected equipment and dispatch the maintenance team immediately '
        'to prevent equipment damage or safety incidents.'
        '</div>',
        unsafe_allow_html=True,
    )
elif latest_risk >= warn_threshold:
    st.markdown(
        '<div class="alert-box alert-warning">'
        '⚠️ <strong>WARNING — Elevated Risk Detected</strong>'
        '<br><br>'
        'Sensor readings are above the warning threshold '
        f'(risk score: <strong>{latest_risk:.0f}</strong> / '
        f'threshold: <strong>{warn_threshold}</strong>).'
        '<br><br>'
        'Schedule a maintenance check within the next few days to '
        'prevent further deterioration.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="alert-box alert-ok">'
        '✅ <strong>All Clear — System Operating Normally</strong>'
        '<br><br>'
        f'Current risk score is <strong>{latest_risk:.0f}</strong>, '
        f'well below the warning threshold of <strong>{warn_threshold}</strong>. '
        'No action is needed at this time.'
        '</div>',
        unsafe_allow_html=True,
    )

st.write("")  # spacer

# ── 2. Anomaly alert ──
if anomaly_count > 0:
    recent = df_filtered[df_filtered["anomaly_label"] == "Anomaly"].tail(3)
    detail_lines = "<br>".join(
        f"&nbsp;&nbsp;• Temp: <strong>{r['temperature']}°C</strong> &nbsp;|&nbsp; "
        f"Pressure: <strong>{r['pressure']} psi</strong> &nbsp;|&nbsp; "
        f"Vibration: <strong>{r['vibration']} mm/s</strong>"
        for _, r in recent.iterrows()
    )
    st.markdown(
        f'<div class="alert-box alert-warning">'
        f'🔍 <strong>Anomaly Detection — {anomaly_count} anomalies found</strong>'
        f'<br><br>'
        f'The AI model flagged <strong>{anomaly_count}</strong> readings '
        f'({round(anomaly_count / max(1, total_readings) * 100, 1)}% of data) '
        f'as abnormal spikes.'
        f'<br><br>'
        f'<strong>Most recent anomalous readings:</strong><br>'
        f'{detail_lines}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.write("")  # spacer

# ── 3. Maintenance overdue alert ──
overdue = [c for c in comp_summary if c["Days Since Maintenance"] > maint_days_warn]
if overdue:
    overdue_lines = "<br>".join(
        f"&nbsp;&nbsp;• <strong>{c['Component']}</strong> — "
        f"last serviced <strong>{c['Days Since Maintenance']}</strong> days ago "
        f"(last: {c['Last Maintenance']})"
        for c in overdue
    )
    st.markdown(
        f'<div class="alert-box alert-warning">'
        f'🔧 <strong>Maintenance Overdue — {len(overdue)} component(s)</strong>'
        f'<br><br>'
        f'The following components have exceeded the '
        f'<strong>{maint_days_warn}-day</strong> service interval:'
        f'<br><br>'
        f'{overdue_lines}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.write("")  # spacer

# ── 4. Aging equipment alert ──
aging_alerts = [c for c in comp_summary if c["Remaining Life %"] < 25]
if aging_alerts:
    aging_lines = "<br>".join(
        f"&nbsp;&nbsp;• <strong>{c['Component']}</strong> — "
        f"{c['Remaining Life %']}% remaining life "
        f"({c['Age (Years)']} yrs of {c['Lifespan (Years)']} yr lifespan) "
        f"— Status: <strong>{c['Aging Status']}</strong>"
        for c in aging_alerts
    )
    st.markdown(
        f'<div class="alert-box alert-warning">'
        f'⏳ <strong>Aging Equipment — {len(aging_alerts)} component(s)</strong>'
        f'<br><br>'
        f'These components are approaching end-of-life and should be '
        f'evaluated for replacement:'
        f'<br><br>'
        f'{aging_lines}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT COMPARISON MODE
# ═══════════════════════════════════════════════════════════════════════

if compare_mode and comp_a and comp_b and comp_a != comp_b:
    st.markdown(
        f'<p class="section-header">⚖️ Comparing: {comp_a} vs {comp_b}</p>',
        unsafe_allow_html=True)

    info_a = next(c for c in comp_summary if c["Component"] == comp_a)
    info_b = next(c for c in comp_summary if c["Component"] == comp_b)
    risk_a = component_risks.get(comp_a, 0)
    risk_b = component_risks.get(comp_b, 0)
    cost_a = estimate_maintenance_cost(comp_a, risk_a)
    cost_b = estimate_maintenance_cost(comp_b, risk_b)
    dtf_a  = estimate_days_to_failure(comp_a, risk_a)
    dtf_b  = estimate_days_to_failure(comp_b, risk_b)

    col_left, col_right = st.columns(2)

    for col, name, info, risk, cost, dtf in [
        (col_left,  comp_a, info_a, risk_a, cost_a, dtf_a),
        (col_right, comp_b, info_b, risk_b, cost_b, dtf_b),
    ]:
        rc = "#00c853" if risk < 35 else "#ff9100" if risk < 65 else "#ff1744"
        with col:
            st.markdown(f"""
            <div class="comp-card">
                <h4>🔩 {name}
                    <span class="status-badge" style="float:right; font-size:.72rem;
                          padding:.18rem .6rem; background:{info['_color']}22;
                          color:{info['_color']}; border:1px solid {info['_color']};">
                          {info['Aging Status']}</span>
                </h4>
                <div class="detail-row"><span class="detail-label">Type</span>
                    <span class="detail-value">{info['Type']}</span></div>
                <div class="detail-row"><span class="detail-label">Age / Lifespan</span>
                    <span class="detail-value">{info['Age (Years)']}yr / {info['Lifespan (Years)']}yr</span></div>
                <div class="detail-row"><span class="detail-label">Remaining Life</span>
                    <span class="detail-value" style="color:{info['_color']};">{info['Remaining Life %']}%</span></div>
                <div class="detail-row"><span class="detail-label">Risk Score</span>
                    <span class="detail-value" style="color:{rc};">{risk}</span></div>
                <div class="detail-row"><span class="detail-label">Est. Days to Failure</span>
                    <span class="detail-value">{dtf:,} days</span></div>
                <div class="detail-row"><span class="detail-label">Maint Cost</span>
                    <span class="detail-value">${cost['cost']:,} ({cost['urgency']})</span></div>
                <div class="detail-row"><span class="detail-label">Criticality</span>
                    <span class="detail-value">{info.get('Criticality','Medium')}</span></div>
            </div>""", unsafe_allow_html=True)

    # Side-by-side sensor charts
    df_a = comp_dfs.get(comp_a, pd.DataFrame())
    df_b = comp_dfs.get(comp_b, pd.DataFrame())
    if not df_a.empty and not df_b.empty:
        for sensor, label in [("temperature","Temperature (°C)"),
                               ("pressure","Pressure (psi)"),
                               ("vibration","Vibration (mm/s)")]:
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(
                    plot_sensor_timeseries(df_a, sensor, title=f"{comp_a} — {label}"),
                    width='stretch')
            with cb:
                st.plotly_chart(
                    plot_sensor_timeseries(df_b, sensor, title=f"{comp_b} — {label}"),
                    width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT INSPECTOR / SYSTEM OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

if selected_component != "🌐 System Overview":
    # ────── Single Component Deep-Dive ──────
    comp_name = selected_component
    st.markdown(
        f'<p class="section-header">🔍 Component Inspector — {comp_name}</p>',
        unsafe_allow_html=True)

    c_info = next(c for c in comp_summary if c["Component"] == comp_name)
    c_risk = component_risks.get(comp_name, 0)
    c_df   = comp_dfs.get(comp_name, pd.DataFrame())
    c_cost = estimate_maintenance_cost(comp_name, c_risk)
    c_dtf  = estimate_days_to_failure(comp_name, c_risk)

    rc = "#00c853" if c_risk < warn_threshold else "#ff9100" if c_risk < crit_threshold else "#ff1744"

    col_card, col_right = st.columns([3, 2])

    with col_card:
        st.markdown(f"""
        <div class="comp-card">
            <h4>🔩 {c_info["Component"]}
                <span class="status-badge" style="float:right; font-size:.72rem;
                      padding:.18rem .6rem; background:{c_info['_color']}22;
                      color:{c_info['_color']}; border:1px solid {c_info['_color']};">
                      {c_info['Aging Status']}</span>
            </h4>
            <div class="detail-row"><span class="detail-label">Type</span>
                <span class="detail-value">{c_info['Type']}</span></div>
            <div class="detail-row"><span class="detail-label">Manufacturer / Model</span>
                <span class="detail-value">{c_info['Manufacturer']} — {c_info['Model']}</span></div>
            <div class="detail-row"><span class="detail-label">Power Rating</span>
                <span class="detail-value">{c_info['Power Rating']}</span></div>
            <div class="detail-row"><span class="detail-label">Installed</span>
                <span class="detail-value">{c_info['Install Date']}</span></div>
            <div class="detail-row"><span class="detail-label">Age</span>
                <span class="detail-value">{c_info['Age (Years)']} yrs of {c_info['Lifespan (Years)']} yr lifespan</span></div>
            <div class="detail-row"><span class="detail-label">Remaining Life</span>
                <span class="detail-value" style="color:{c_info['_color']};">{c_info['Remaining Life %']}%</span></div>
            <div class="detail-row"><span class="detail-label">Last Maintenance</span>
                <span class="detail-value">{c_info['Last Maintenance']} ({c_info['Days Since Maintenance']}d ago)</span></div>
            <div class="detail-row"><span class="detail-label">Criticality</span>
                <span class="detail-value">{c_info.get('Criticality','Medium')}</span></div>
            <div class="detail-row"><span class="detail-label">Risk Score</span>
                <span class="detail-value" style="color:{rc};">{c_risk}</span></div>
            <div style="color:#64748b; font-size:.8rem; margin-top:.4rem; font-style:italic;">
                {c_info['Description']}</div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.plotly_chart(plot_risk_gauge(c_risk), width='stretch')

        if show_failure_timeline:
            dtf_color = "#ff1744" if c_dtf < 365 else "#ff9100" if c_dtf < 730 else "#00c853"
            st.markdown(f"""
            <div class="cost-card">
                <h4>⏱️ Failure Prediction</h4>
                <div class="cost-row"><span class="cost-label">Est. Days to Failure</span>
                    <span class="cost-value" style="color:{dtf_color};">{c_dtf:,} days</span></div>
                <div class="cost-row"><span class="cost-label">Approx. Date</span>
                    <span class="cost-value">{(datetime.now() + pd.Timedelta(days=c_dtf)).strftime('%b %Y')}</span></div>
            </div>""", unsafe_allow_html=True)

        if show_cost_analysis:
            st.markdown(f"""
            <div class="cost-card">
                <h4>💰 Cost Analysis</h4>
                <div class="cost-row"><span class="cost-label">Recommended</span>
                    <span class="cost-value">{c_cost['action']}</span></div>
                <div class="cost-row"><span class="cost-label">Estimated Cost</span>
                    <span class="cost-value">${c_cost['cost']:,}</span></div>
                <div class="cost-row"><span class="cost-label">Urgency</span>
                    <span class="cost-value" style="color:{c_cost['urgency_color']};">{c_cost['urgency']}</span></div>
            </div>""", unsafe_allow_html=True)

    # Component sensor charts
    if not c_df.empty:
        chart_df = c_df[c_df["anomaly_label"] == "Anomaly"] if show_anomalies_only else c_df
        st.markdown(f'<p class="section-header">📈 {comp_name} — Sensor Readings</p>',
                    unsafe_allow_html=True)
        tab_t, tab_p, tab_v = st.tabs(["🌡️ Temperature","🔵 Pressure","📳 Vibration"])
        with tab_t:
            st.plotly_chart(plot_sensor_timeseries(chart_df, "temperature",
                            title=f"{comp_name} — Temperature (°C)"), width='stretch')
        with tab_p:
            st.plotly_chart(plot_sensor_timeseries(chart_df, "pressure",
                            title=f"{comp_name} — Pressure (psi)"), width='stretch')
        with tab_v:
            st.plotly_chart(plot_sensor_timeseries(chart_df, "vibration",
                            title=f"{comp_name} — Vibration (mm/s)"), width='stretch')

else:
    # ────── System Overview ──────
    st.markdown('<p class="section-header">🏭 Component Inventory</p>',
                unsafe_allow_html=True)

    for i in range(0, len(comp_summary), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(comp_summary):
                break
            c = comp_summary[idx]
            rv = component_risks.get(c["Component"], 0)
            rc = "#00c853" if rv < warn_threshold else "#ff9100" if rv < crit_threshold else "#ff1744"
            dtf = estimate_days_to_failure(c["Component"], rv)
            dtf_c = "#ff1744" if dtf < 365 else "#ff9100" if dtf < 730 else "#00c853"
            with col:
                st.markdown(f"""
                <div class="comp-card">
                    <h4>🔩 {c["Component"]}
                        <span class="status-badge" style="float:right; font-size:.7rem;
                              padding:.15rem .5rem; background:{c['_color']}22;
                              color:{c['_color']}; border:1px solid {c['_color']};">
                              {c['Aging Status']}</span>
                    </h4>
                    <div class="detail-row"><span class="detail-label">Type</span>
                        <span class="detail-value">{c['Type']}</span></div>
                    <div class="detail-row"><span class="detail-label">Manufacturer</span>
                        <span class="detail-value">{c['Manufacturer']}</span></div>
                    <div class="detail-row"><span class="detail-label">Age / Lifespan</span>
                        <span class="detail-value">{c['Age (Years)']}yr / {c['Lifespan (Years)']}yr</span></div>
                    <div class="detail-row"><span class="detail-label">Remaining Life</span>
                        <span class="detail-value" style="color:{c['_color']};">{c['Remaining Life %']}%</span></div>
                    <div class="detail-row"><span class="detail-label">Risk</span>
                        <span class="detail-value" style="color:{rc};">{rv}</span></div>
                    <div class="detail-row"><span class="detail-label">Days to Failure</span>
                        <span class="detail-value" style="color:{dtf_c};">{dtf:,}d</span></div>
                    <div class="detail-row"><span class="detail-label">Criticality</span>
                        <span class="detail-value">{c.get('Criticality','Medium')}</span></div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# COST ANALYSIS SUMMARY (when in system overview)
# ═══════════════════════════════════════════════════════════════════════

if selected_component == "🌐 System Overview" and show_cost_analysis:
    st.markdown('<p class="section-header">💰 Maintenance Cost Estimator</p>',
                unsafe_allow_html=True)

    cost_cols = st.columns(3)
    for i, comp in enumerate(DEFAULT_COMPONENTS):
        rv = component_risks.get(comp, 0)
        cost = estimate_maintenance_cost(comp, rv)
        dtf = estimate_days_to_failure(comp, rv)
        with cost_cols[i % 3]:
            st.markdown(f"""
            <div class="cost-card">
                <h4>🔩 {comp}</h4>
                <div class="cost-row"><span class="cost-label">Action</span>
                    <span class="cost-value">{cost['action']}</span></div>
                <div class="cost-row"><span class="cost-label">Cost</span>
                    <span class="cost-value">${cost['cost']:,}</span></div>
                <div class="cost-row"><span class="cost-label">Urgency</span>
                    <span class="cost-value" style="color:{cost['urgency_color']};">{cost['urgency']}</span></div>
                <div class="cost-row"><span class="cost-label">Days to Failure</span>
                    <span class="cost-value">{dtf:,}d</span></div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# AGING & MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">⏳ Component Aging & Maintenance</p>',
            unsafe_allow_html=True)

col_ag, col_mt = st.columns(2)
with col_ag:
    st.plotly_chart(plot_aging_chart(comp_summary), width='stretch')
with col_mt:
    st.plotly_chart(plot_maintenance_timeline(comp_summary), width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM-WIDE SENSOR CHARTS
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">📈 System-Wide Sensor Readings</p>',
            unsafe_allow_html=True)

chart_df = df_filtered[df_filtered["anomaly_label"] == "Anomaly"] if show_anomalies_only else df_filtered

if show_anomalies_only:
    st.markdown('<div class="alert-box alert-info">🔎 Showing <strong>anomalies only</strong>.</div>',
                unsafe_allow_html=True)

tab_t, tab_p, tab_v = st.tabs(["🌡️ Temperature","🔵 Pressure","📳 Vibration"])
with tab_t:
    st.plotly_chart(plot_sensor_timeseries(chart_df, "temperature", title="Temperature (°C)"), width='stretch')
with tab_p:
    st.plotly_chart(plot_sensor_timeseries(chart_df, "pressure", title="Pressure (psi)"), width='stretch')
with tab_v:
    st.plotly_chart(plot_sensor_timeseries(chart_df, "vibration", title="Vibration (mm/s)"), width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">🎯 Risk Analysis</p>', unsafe_allow_html=True)

col_g, col_h = st.columns([1, 2])
with col_g:
    st.plotly_chart(plot_risk_gauge(latest_risk), width='stretch')
with col_h:
    st.plotly_chart(plot_risk_heatmap(list(component_risks.keys()),
                    list(component_risks.values())), width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">🕸️ System Dependency & Failure Propagation</p>',
            unsafe_allow_html=True)

graph = build_system_graph()
graph = assign_risk_to_nodes(graph, component_risks)

if simulate_fail and failed_component:
    graph = simulate_failure_propagation(graph, failed_component,
                                        propagation_factor=propagation_factor)
    st.markdown(
        f'<div class="alert-box alert-critical">'
        f'💥 <strong>{failed_component}</strong> failed — cascade strength '
        f'{propagation_factor:.0%}.</div>', unsafe_allow_html=True)

st.plotly_chart(plot_network_graph(graph), width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# DATA TABLE & DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

if show_data_table:
    st.markdown('<p class="section-header">📋 Data Explorer</p>', unsafe_allow_html=True)
    disp = df_filtered[["timestamp","temperature","pressure","vibration",
                         "anomaly_label","risk_score","failure_prob"]].copy()
    disp.columns = ["Timestamp","Temp (°C)","Press (psi)","Vib (mm/s)",
                     "Status","Risk","Fail%"]
    st.dataframe(disp, width='stretch', height=350)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Full Report (CSV)", data=csv,
                   file_name="predictive_maintenance_report.csv",
                   mime="text/csv", width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:.8rem;">'
    '🔧 AI Predictive Maintenance Dashboard · '
    'Powered by Isolation Forest & Random Forest · '
    'Built with Streamlit & Plotly</p>',
    unsafe_allow_html=True)
