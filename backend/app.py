"""
NexusGuard — Unified Predictive Maintenance Platform
=====================================================
NEXUS × InfraGuard Fusion | DS-PS2 Submission

TWO WAYS TO RUN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Standalone HTML (no install needed):
   Just open  dashboard.html  in any browser.
   Full interactive dashboard with light/dark toggle.

2. Streamlit app (Python backend):
   pip install -r requirements.txt
   streamlit run app.py
   → http://localhost:8501
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from backend.topology import COMPONENTS, COMPONENT_DETAILS, EDGES
from backend.data_generator import (
    generate_component_data, generate_realtime_point,
    get_remaining_life_pct, get_component_age_years, get_days_since_maintenance,
)
from backend.models import (
    train_anomaly_detector, detect_anomalies,
    calculate_risk_score, calculate_risk_scores,
    train_failure_predictor, predict_failure,
    get_system_status, compute_infraguard_rs,
)
from backend.graph_model import (
    build_system_graph, assign_risk_to_nodes, assign_rs_to_nodes,
    simulate_failure_propagation, build_component_summary,
    estimate_days_to_failure, estimate_maintenance_cost,
    get_system_health_score, generate_causal_maintenance_recommendations,
)
from frontend.visualizations import (
    plot_sensor_timeseries, plot_risk_heatmap, plot_rs_breakdown,
    plot_network_graph, plot_risk_gauge, plot_rs_gauge,
    plot_aging_chart, plot_cascade_impact, plot_maintenance_timeline,
)

st.set_page_config(
    page_title="NexusGuard — Predictive Maintenance",
    page_icon="⬡", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
.main-header {
    background: linear-gradient(135deg,#070b14 0%,#0c1220 50%,#0f1e3a 100%);
    padding:1.4rem 2rem; border-radius:14px; margin-bottom:1.2rem;
    border:1px solid rgba(0,212,170,0.2);
}
.main-header h1 { color:#e8edf5; font-weight:700; margin:0 0 .2rem; font-size:1.75rem; }
.main-header p  { color:#6b7a99; margin:0; font-size:.88rem; }
.section-hdr { color:#00d4aa; font-size:.95rem; font-weight:700;
    margin:1.4rem 0 .5rem; padding-bottom:.3rem;
    border-bottom:2px solid rgba(0,212,170,.2); letter-spacing:.05em; }
.alert-critical { background:rgba(255,59,92,.12); border-left:4px solid #ff3b5c;
    color:#ff7096; padding:.7rem 1rem; border-radius:8px; margin:.4rem 0; font-size:.88rem; }
.alert-warning  { background:rgba(245,166,35,.10); border-left:4px solid #f5a623;
    color:#ffc666; padding:.7rem 1rem; border-radius:8px; margin:.4rem 0; font-size:.88rem; }
.alert-ok       { background:rgba(31,224,144,.10); border-left:4px solid #1fe090;
    color:#69f0ae; padding:.7rem 1rem; border-radius:8px; margin:.4rem 0; font-size:.88rem; }
.rec-card { background:linear-gradient(145deg,#0c1220,#0a1018);
    padding:.9rem 1.1rem; border-radius:10px; border:1px solid #1a2540; margin-bottom:.5rem; }
.rec-card.p1 { border-left:3px solid #ff3b5c; }
.rec-card.p2 { border-left:3px solid #f5a623; }
.rec-card.p3 { border-left:3px solid #6366f1; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#070b14 0%,#0c1220 100%); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ NexusGuard")
    st.markdown('<p style="color:#6b7a99;font-size:.8rem;margin-top:-.5rem">NEXUS × InfraGuard</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**📊 Data Simulation**")
    n_points      = st.slider("Data points",  100, 1000, 400, 50)
    anomaly_ratio = st.slider("Anomaly ratio", 0.01, 0.25, 0.06, 0.01)
    st.divider()
    st.markdown("**🔬 ML Config**")
    contamination  = st.slider("IsoForest contamination", 0.01, 0.20, 0.06, 0.01)
    risk_threshold = st.slider("RF failure threshold",    40,   80,   62,   1)
    st.divider()
    st.markdown("**🔥 Cascade Simulation**")
    sim_component = st.selectbox("Seed node", COMPONENTS)
    stressor_type = st.selectbox("Stressor", ["load_surge","thermal","compound"], index=2)
    delta_load    = st.slider("Load surge Δ", 0.1, 1.0, 0.5, 0.05)
    delta_temp    = st.slider("Temp spike Δ°C", 5.0, 40.0, 20.0, 1.0)
    run_sim       = st.button("🔥 Simulate Cascade", type="primary", use_container_width=True)
    st.divider()
    auto_refresh  = st.checkbox("Auto-refresh")
    refresh_secs  = st.slider("Interval (s)", 2, 15, 4) if auto_refresh else 4
    if st.button("↺ Regenerate", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    selected_comp = st.selectbox("Inspect component", COMPONENTS)

# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=refresh_secs if auto_refresh else 3600)
def load_data(n, ar, cont, rt):
    comp_data = generate_component_data(COMPONENTS, n_points=n, anomaly_ratio=ar)
    all_df    = pd.concat(comp_data.values(), ignore_index=True)
    iso_m, iso_s = train_anomaly_detector(all_df, contamination=cont)
    rf_m,  rf_s  = train_failure_predictor(all_df, risk_threshold=rt)
    component_risks = {}
    for comp, df in comp_data.items():
        df["anomaly_label"] = detect_anomalies(iso_m, iso_s, df)
        df["risk_score"]    = calculate_risk_scores(df)
        pf = predict_failure(rf_m, rf_s, df)
        df["failure_prob"]  = pf["failure_prob"]
        component_risks[comp] = float(df["risk_score"].iloc[-1])
        comp_data[comp] = df
    component_ages = {c: COMPONENT_DETAILS[c].get("age",5) for c in COMPONENTS}
    rs_data = compute_infraguard_rs(component_risks, EDGES, component_ages)
    G = build_system_graph()
    G = assign_risk_to_nodes(G, component_risks)
    G = assign_rs_to_nodes(G, rs_data)
    return comp_data, component_risks, rs_data, G

comp_data, component_risks, rs_data, G = load_data(n_points, anomaly_ratio, contamination, risk_threshold)

cascade_log, sim_graph = [], G.copy()
if run_sim or st.session_state.get("sim_ran"):
    if run_sim:
        st.session_state.update({"sim_ran":True,"sim_comp":sim_component,"sim_st":stressor_type,"sim_dl":delta_load,"sim_dt":delta_temp})
    component_loads = {c: component_risks[c] for c in COMPONENTS}
    sim_graph, cascade_log = simulate_failure_propagation(
        G.copy(), st.session_state.get("sim_comp", sim_component),
        component_loads=component_loads,
        stressor_type=st.session_state.get("sim_st", stressor_type),
        delta_load=st.session_state.get("sim_dl", delta_load),
        delta_temp=st.session_state.get("sim_dt", delta_temp),
    )

health_score  = get_system_health_score(component_risks)
avg_rs        = np.mean([d["Rs"] for d in rs_data.values()])
critical_list = [c for c in COMPONENTS if component_risks[c] >= 65]
warning_list  = [c for c in COMPONENTS if 35 <= component_risks[c] < 65]
recommendations = generate_causal_maintenance_recommendations(component_risks, rs_data, G)

# ── Header ────────────────────────────────────────────────────────────────────
sys_status, sys_color = get_system_status(max(component_risks.values()))
st.markdown(f"""
<div class="main-header">
  <h1>⬡ NexusGuard Command Center</h1>
  <p>NEXUS × InfraGuard &nbsp;|&nbsp; {len(COMPONENTS)} components &nbsp;|&nbsp;
     {len(EDGES)} PTDF edges &nbsp;|&nbsp;
     Updated {datetime.now().strftime('%H:%M:%S')}</p>
</div>""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
m1,m2,m3,m4,m5,m6 = st.columns(6)
for col, lbl, val, c in [
    (m1,"Status",        sys_status,                    sys_color),
    (m2,"Health",        f"{health_score:.0f}/100",      "#00d4aa"),
    (m3,"Avg Rs",        f"{avg_rs:.3f}",               "#a78bfa"),
    (m4,"Critical",      f"{len(critical_list)}",        "#ff3b5c"),
    (m5,"At Risk",       f"{len(warning_list)}",         "#f5a623"),
    (m6,"Cascade",       f"{len(cascade_log)} steps" if cascade_log else "No sim", "#6366f1"),
]:
    col.markdown(f"""<div style="background:#0c1220;border:1px solid #1a2540;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:9px;color:#6b7a99;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">{lbl}</div>
        <div style="font-size:18px;font-weight:700;color:{c};font-family:'JetBrains Mono',monospace">{val}</div>
    </div>""", unsafe_allow_html=True)

# ── Alerts ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Live Alerts</div>', unsafe_allow_html=True)
a1, a2 = st.columns(2)
with a1:
    for comp in critical_list:
        st.markdown(f'<div class="alert-critical">🚨 <b>{comp}</b> — Risk {component_risks[comp]:.0f}/100 · Rs={rs_data[comp]["Rs"]:.3f}</div>', unsafe_allow_html=True)
    if not critical_list:
        st.markdown('<div class="alert-ok">✅ No critical components</div>', unsafe_allow_html=True)
with a2:
    for comp in warning_list[:4]:
        st.markdown(f'<div class="alert-warning">⚠️ <b>{comp}</b> — Risk {component_risks[comp]:.0f}/100 · Rs={rs_data[comp]["Rs"]:.3f}</div>', unsafe_allow_html=True)

# ── Network + Heatmap ─────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">System Topology</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns([1.5, 1])
with r1c1:
    st.plotly_chart(plot_network_graph(sim_graph if cascade_log else G, rs_data), use_container_width=True)
    if cascade_log:
        st.plotly_chart(plot_cascade_impact(cascade_log, COMPONENTS), use_container_width=True)
with r1c2:
    st.plotly_chart(plot_risk_gauge(max(component_risks.values())), use_container_width=True)
    st.plotly_chart(plot_risk_heatmap(component_risks), use_container_width=True)

# ── Anomaly trends ────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-hdr">Anomaly Trends — {selected_comp}</div>', unsafe_allow_html=True)
df_sel = comp_data[selected_comp]
tc1, tc2, tc3 = st.columns(3)
with tc1: st.plotly_chart(plot_sensor_timeseries(df_sel,"temperature",selected_comp), use_container_width=True)
with tc2: st.plotly_chart(plot_sensor_timeseries(df_sel,"pressure",   selected_comp), use_container_width=True)
with tc3: st.plotly_chart(plot_sensor_timeseries(df_sel,"vibration",  selected_comp), use_container_width=True)

# ── Rs breakdown ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">InfraGuard Rs Decomposition</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns([2, 1])
with rc1: st.plotly_chart(plot_rs_breakdown(rs_data), use_container_width=True)
with rc2: st.plotly_chart(plot_rs_gauge(rs_data[selected_comp]["Rs"], selected_comp), use_container_width=True)

# ── Maintenance ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Causal Maintenance Recommendations</div>', unsafe_allow_html=True)
mc1, mc2 = st.columns([1.4, 1])
with mc1:
    pc = {1:"#ff3b5c",2:"#f5a623",3:"#6366f1"}
    pl = {1:"P1 · 24h",2:"P2 · 7d",3:"P3 · sched"}
    for rec in recommendations[:6]:
        p = rec["priority"]
        trig = f" ← <b style='color:{pc[p]}'>{rec['triggered_by']}</b>" if rec["triggered_by"] else ""
        st.markdown(f"""<div class="rec-card p{p}">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:{pc[p]};font-weight:700;font-size:.78rem">{pl[p]}</span>
                <span style="color:#6b7a99;font-size:.72rem">Rs={rec['Rs']:.3f}</span>
            </div>
            <div style="color:#e8edf5;font-size:.88rem;font-weight:600">{rec['action']} — {rec['component']}{trig}</div>
            <div style="color:#6b7a99;font-size:.75rem">{rec['days_to_failure']} days · ${rec['cost']:,} · {rec['remaining_life_pct']:.0f}% life</div>
            <div style="color:#4a5568;font-size:.72rem;margin-top:2px">{rec['causal_chain'][-1]}</div>
        </div>""", unsafe_allow_html=True)
with mc2:
    summary = build_component_summary()
    st.plotly_chart(plot_aging_chart(summary), use_container_width=True)
    st.plotly_chart(plot_maintenance_timeline(recommendations), use_container_width=True)

# ── Component table ───────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Component Registry</div>', unsafe_allow_html=True)
rows = []
for comp in COMPONENTS:
    info = COMPONENT_DETAILS[comp]
    cost = estimate_maintenance_cost(comp, component_risks[comp])
    rows.append({
        "Component": comp, "Type": info["type"],
        "Risk /100":  f"{component_risks[comp]:.0f}",
        "Rs":         f"{rs_data[comp]['Rs']:.3f}",
        "P_fail":     f"{rs_data[comp]['Pf']:.3f}",
        "Load %":     f"{info['P_rated']:.0f}%",
        "Age (yr)":   get_component_age_years(comp),
        "Life %":     f"{get_remaining_life_pct(comp):.0f}%",
        "Action":     cost["action"],
        "Est. Cost":  f"${cost['cost']:,}",
        "Urgency":    cost["urgency"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
e1, e2 = st.columns(2)
all_export = pd.concat([df.assign(component=c) for c,df in comp_data.items()], ignore_index=True)
with e1:
    st.download_button("⬇ Sensor Report (CSV)", all_export.to_csv(index=False).encode(),
                       "nexusguard_sensors.csv","text/csv",use_container_width=True)
with e2:
    rec_df = pd.DataFrame([{"Priority":r["priority"],"Component":r["component"],
        "Action":r["action"],"Rs":r["Rs"],"Days":r["days_to_failure"],
        "Cost":r["cost"],"Triggered By":r["triggered_by"] or "—"} for r in recommendations])
    st.download_button("⬇ Maintenance Report (CSV)", rec_df.to_csv(index=False).encode(),
                       "nexusguard_maintenance.csv","text/csv",use_container_width=True)

if auto_refresh:
    import time; time.sleep(refresh_secs); st.rerun()
