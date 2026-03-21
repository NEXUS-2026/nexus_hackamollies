"""
StruXel — Where structures speak before they break.
=====================================================
Predictive Maintenance Platform | DS-PS2 Submission

HOW TO RUN:
  pip install -r requirements.txt
  streamlit run struxel_app.py
  → http://localhost:8501
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
    page_title="StruXel — Where structures speak before they break.",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #f4f2ee; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    st.markdown("""
    <style>
    .stApp { background: #f4f2ee !important; }
    .block-container { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── NAV ──
    col_logo, col_links, col_cta = st.columns([2, 4, 2])
    with col_logo:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:18px 0 18px 32px">
            <div style="width:18px;height:18px;background:#c9922a;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);flex-shrink:0"></div>
            <span style="font-size:18px;font-weight:800;color:#1a1a18;letter-spacing:-.02em">StruXel</span>
        </div>
        """, unsafe_allow_html=True)
    with col_links:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;gap:28px;padding:18px 0">
            <a href="#features" style="font-size:13px;color:#888;text-decoration:none">Features</a>
            <a href="https://www.researchgate.net/publication/220565847_Anomaly_Detection_A_Survey" target="_blank" style="font-size:13px;color:#888;text-decoration:none">Docs</a>
        </div>
        """, unsafe_allow_html=True)
    with col_cta:
        st.markdown('<div style="padding:14px 32px 14px 0;text-align:right">', unsafe_allow_html=True)
        if st.button("Get started →", key="nav_cta", type="primary"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:0;border:none;border-top:1px solid #e0ddd8">', unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div style="text-align:center;padding:80px 48px 56px;background:#f4f2ee;position:relative">
        <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-bottom:28px">
            <span style="font-size:10px;padding:4px 12px;border-radius:20px;font-weight:600;font-family:'JetBrains Mono',monospace;background:#fdeaea;color:#b83030;border:1px solid #f0b0b0">IsolationForest</span>
            <span style="font-size:10px;padding:4px 12px;border-radius:20px;font-weight:600;font-family:'JetBrains Mono',monospace;background:#fdf3e3;color:#9a6800;border:1px solid #f0d090">PTDF Cascade</span>
            <span style="font-size:10px;padding:4px 12px;border-radius:20px;font-weight:600;font-family:'JetBrains Mono',monospace;background:#e8f5e8;color:#2d7a2d;border:1px solid #b8ddb8">Rs Scoring</span>
        </div>
        <h1 style="font-size:clamp(32px,4vw,56px);font-weight:800;color:#1a1a18;line-height:1.1;letter-spacing:-.03em;margin-bottom:18px">
            Where <span style="color:#c9922a">structures speak</span><br>before they break.
        </h1>
        <p style="font-size:15px;color:#888;max-width:460px;margin:0 auto 32px;line-height:1.75">
            Predictive maintenance powered by graph-based risk scoring, cascade simulation and real-time anomaly detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_l, cta_c, cta_r = st.columns([3, 2, 3])
    with cta_c:
        if st.button("Open dashboard", key="hero_cta", type="primary", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    # ── STATS STRIP ──
    st.markdown('<hr style="margin:32px 0 0;border:none;border-top:1px solid #e0ddd8">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for col, num, label, color in [
        (s1, "14.2h",  "Downtime prevented",      "#c9922a"),
        (s2, "₹1.8Cr", "Economic impact saved",   "#4e9e4e"),
        (s3, "2",       "Cascade events blocked",  "#6496e0"),
    ]:
        col.markdown(f"""
        <div style="text-align:center;padding:28px 20px;background:#f0ede8;border-right:1px solid #e0ddd8">
            <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{color};margin-bottom:6px">{num}</div>
            <div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.14em;font-weight:600">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── FEATURES SECTION ──
    st.markdown('<hr style="margin:0;border:none;border-top:1px solid #e0ddd8">', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:64px 48px 0">
        <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#c9922a;font-family:'JetBrains Mono',monospace;margin-bottom:36px;display:flex;align-items:center;gap:12px">
            Core capabilities
            <span style="flex:1;height:1px;background:#e0ddd8;display:inline-block"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    features = [
        ("#c9922a", "#fdf3e3", "01", "Rs risk scoring",
         "Neighbourhood-aware composite scoring using failure probability, centrality and impact — updated in real-time.",
         "Rs = 0.40·Pf + 0.35·C + 0.25·I", "#fdf3e3", "#9a6800"),
        ("#e05252", "#fdeaea", "02", "Cascade simulation",
         "PTDF-based deterministic load redistribution engine. Simulate compound, thermal or load surge failures.",
         "PTDF · Physics-based", "#fdeaea", "#b83030"),
        ("#4e9e4e", "#e8f5e8", "03", "Anomaly detection",
         "Isolation Forest per component with Weibull and LogNormal sensor simulation across temperature, pressure and vibration.",
         "IsolationForest · ML", "#e8f5e8", "#2d7a2d"),
    ]
    for col, (color, bg, num, title, desc, tag, tag_bg, tag_c) in zip([f1, f2, f3], features):
        col.markdown(f"""
        <div style="background:#ffffff;padding:28px;border:1px solid #e0ddd8;border-top:2px solid {color};margin:0 0 24px">
            <div style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#ccc;margin-bottom:10px">{num}</div>
            <div style="width:36px;height:36px;border-radius:10px;background:{bg};display:flex;align-items:center;justify-content:center;margin-bottom:14px">
                <div style="width:10px;height:10px;border-radius:50%;background:{color}"></div>
            </div>
            <div style="font-size:15px;font-weight:600;color:#1a1a18;margin-bottom:8px">{title}</div>
            <div style="font-size:12px;color:#888;line-height:1.7;margin-bottom:12px">{desc}</div>
            <span style="font-size:9px;padding:3px 10px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-weight:600;background:{tag_bg};color:{tag_c}">{tag}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── HOW IT WORKS ──
    st.markdown("""
    <div style="padding:0 48px">
        <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#c9922a;font-family:'JetBrains Mono',monospace;margin-bottom:36px;display:flex;align-items:center;gap:12px">
            How it works
            <span style="flex:1;height:1px;background:#e0ddd8;display:inline-block"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("#6496e0", "01", "Sensor ingestion", "Temperature, pressure and vibration streamed continuously per component.", "Real-time · IoT", "#e6f1fb", "#185fa5"),
        ("#e05252", "02", "Anomaly detection", "Isolation Forest flags outliers across all sensor dimensions.", "IsolationForest · ML", "#fdeaea", "#b83030"),
        ("#c9922a", "03", "Rs risk scoring", "Composite Rs = 0.40·Pf + 0.35·C + 0.25·I updated each cycle.", "Rs scoring · Live", "#fdf3e3", "#9a6800"),
        ("#4e9e4e", "04", "Recommendation", "Causal chain analysis generates P1/P2/P3 maintenance actions with ₹ cost.", "Causal chain · Priority", "#e8f5e8", "#2d7a2d"),
    ]
    for col, (color, num, title, desc, tag, tag_bg, tag_c) in zip([h1, h2, h3, h4], steps):
        col.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e0ddd8;padding:24px;margin:0 0 24px">
            <div style="font-size:28px;font-weight:800;font-family:'JetBrains Mono',monospace;color:#e0ddd8;margin-bottom:12px">{num}</div>
            <div style="font-size:13px;font-weight:600;color:{color};margin-bottom:6px">{title}</div>
            <div style="font-size:11px;color:#888;line-height:1.7;margin-bottom:10px">{desc}</div>
            <span style="font-size:9px;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-weight:600;background:{tag_bg};color:{tag_c}">{tag}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown('<hr style="margin:0;border:none;border-top:1px solid #e0ddd8">', unsafe_allow_html=True)
    fl, fc, fr = st.columns([2, 4, 2])
    with fl:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:20px 0 20px 32px">
            <div style="width:14px;height:14px;background:#c9922a;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%)"></div>
            <span style="font-size:14px;font-weight:700;color:#1a1a18">StruXel</span>
        </div>
        """, unsafe_allow_html=True)
    with fc:
        st.markdown('<div style="text-align:center;padding:22px 0;font-size:11px;color:#aaa;font-family:\'JetBrains Mono\',monospace">Where structures speak before they break.</div>', unsafe_allow_html=True)
    with fr:
        st.markdown('<div style="padding:14px 32px 14px 0;text-align:right">', unsafe_allow_html=True)
        if st.button("Open dashboard →", key="footer_cta"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    st.markdown("""
    <style>
    .stApp { background: #111110 !important; }
    [class*="css"] { font-family: 'Sora', sans-serif !important; }
    .main-header {
        background: #1e1e1b; padding:1.2rem 2rem; border-radius:14px;
        margin-bottom:1rem; border:1px solid #2e2e2a;
    }
    .main-header h1 { color:#f0ece4; font-weight:700; margin:0; font-size:1.5rem; letter-spacing:-.02em; }
    .section-hdr { color:#c9922a; font-size:.85rem; font-weight:700;
        margin:1.2rem 0 .5rem; padding-bottom:.3rem;
        border-bottom:1px solid #2e2e2a; letter-spacing:.08em; text-transform:uppercase; }
    .alert-critical { background:#2e0f0f; border-left:4px solid #e05252;
        color:#e08080; padding:.7rem 1rem; border-radius:8px; margin:.3rem 0; font-size:.85rem; }
    .alert-warning { background:#2e1f0a; border-left:4px solid #c9922a;
        color:#c9a040; padding:.7rem 1rem; border-radius:8px; margin:.3rem 0; font-size:.85rem; }
    .alert-ok { background:#0a1e0a; border-left:4px solid #4e9e4e;
        color:#60a060; padding:.7rem 1rem; border-radius:8px; margin:.3rem 0; font-size:.85rem; }
    .rec-card { background:#1a1a18; padding:.9rem 1.1rem; border-radius:10px;
        border:1px solid #2e2e2a; margin-bottom:.5rem; }
    .rec-card.p1 { border-left:3px solid #e05252; }
    .rec-card.p2 { border-left:3px solid #c9922a; }
    .rec-card.p3 { border-left:3px solid #6496e0; }
    section[data-testid="stSidebar"] { background:#111110; border-right:1px solid #2e2e2a; }
    .stDataFrame { background:#1e1e1b !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⬡ StruXel")
        st.markdown('<p style="color:#666;font-size:.8rem;margin-top:-.5rem">Predictive Maintenance</p>', unsafe_allow_html=True)
        if st.button("← Back to home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
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

    # ── Data pipeline ──
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
    sys_status, sys_color = get_system_status(max(component_risks.values()))

    # ── Header ──
    st.markdown(f"""
    <div class="main-header">
      <h1>⬡ StruXel</h1>
      <p style="color:#666;margin:2px 0 0;font-size:.85rem;font-family:'JetBrains Mono',monospace">
        Where structures speak before they break. &nbsp;·&nbsp; Updated {datetime.now().strftime('%H:%M:%S')}
      </p>
    </div>""", unsafe_allow_html=True)

    # ── KPI Metrics ──
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    for col, lbl, val, c in [
        (m1,"Status",   sys_status,                   sys_color),
        (m2,"Health",   f"{health_score:.0f}/100",     "#4e9e4e"),
        (m3,"Avg Rs",   f"{avg_rs:.3f}",              "#c9922a"),
        (m4,"Critical", f"{len(critical_list)}",       "#e05252"),
        (m5,"At Risk",  f"{len(warning_list)}",        "#c9922a"),
        (m6,"Cascade",  f"{len(cascade_log)} steps" if cascade_log else "No sim", "#6496e0"),
    ]:
        col.markdown(f"""<div style="background:#1e1e1b;border:1px solid #2e2e2a;border-radius:10px;padding:12px;text-align:center;margin-bottom:10px">
            <div style="font-size:9px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">{lbl}</div>
            <div style="font-size:18px;font-weight:700;color:{c};font-family:'JetBrains Mono',monospace">{val}</div>
        </div>""", unsafe_allow_html=True)

    # ── Alerts ──
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

    # ── Network + Heatmap ──
    st.markdown('<div class="section-hdr">System Topology</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([1.5, 1])
    with r1c1:
        st.plotly_chart(plot_network_graph(sim_graph if cascade_log else G, rs_data), use_container_width=True)
        if cascade_log:
            st.plotly_chart(plot_cascade_impact(cascade_log, COMPONENTS), use_container_width=True)
    with r1c2:
        st.plotly_chart(plot_risk_gauge(max(component_risks.values())), use_container_width=True)
        st.plotly_chart(plot_risk_heatmap(component_risks), use_container_width=True)

    # ── Telemetry ──
    st.markdown(f'<div class="section-hdr">Anomaly Trends — {selected_comp}</div>', unsafe_allow_html=True)
    df_sel = comp_data[selected_comp]
    tc1, tc2, tc3 = st.columns(3)
    with tc1: st.plotly_chart(plot_sensor_timeseries(df_sel,"temperature",selected_comp), use_container_width=True)
    with tc2: st.plotly_chart(plot_sensor_timeseries(df_sel,"pressure",   selected_comp), use_container_width=True)
    with tc3: st.plotly_chart(plot_sensor_timeseries(df_sel,"vibration",  selected_comp), use_container_width=True)

    # ── Rs breakdown ──
    st.markdown('<div class="section-hdr">InfraGuard Rs Decomposition</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([2, 1])
    with rc1: st.plotly_chart(plot_rs_breakdown(rs_data), use_container_width=True)
    with rc2: st.plotly_chart(plot_rs_gauge(rs_data[selected_comp]["Rs"], selected_comp), use_container_width=True)

    # ── Maintenance ──
    st.markdown('<div class="section-hdr">Causal Maintenance Recommendations</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns([1.4, 1])
    with mc1:
        pc = {1:"#e05252",2:"#c9922a",3:"#6496e0"}
        pl = {1:"P1 · 24h",2:"P2 · 7d",3:"P3 · sched"}
        for rec in recommendations[:6]:
            p = rec["priority"]
            trig = f" ← <b style='color:{pc[p]}'>{rec['triggered_by']}</b>" if rec["triggered_by"] else ""
            st.markdown(f"""<div class="rec-card p{p}">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="color:{pc[p]};font-weight:700;font-size:.78rem">{pl[p]}</span>
                    <span style="color:#555;font-size:.72rem">Rs={rec['Rs']:.3f}</span>
                </div>
                <div style="color:#f0ece4;font-size:.88rem;font-weight:600">{rec['action']} — {rec['component']}{trig}</div>
                <div style="color:#666;font-size:.75rem">{rec['days_to_failure']} days · ₹{rec['cost']*83:,.0f} · {rec['remaining_life_pct']:.0f}% life</div>
                <div style="color:#444;font-size:.72rem;margin-top:2px">{rec['causal_chain'][-1]}</div>
            </div>""", unsafe_allow_html=True)
    with mc2:
        summary = build_component_summary()
        st.plotly_chart(plot_aging_chart(summary), use_container_width=True)
        st.plotly_chart(plot_maintenance_timeline(recommendations), use_container_width=True)

    # ── Component table ──
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
            "Age (yr)":   get_component_age_years(comp),
            "Life %":     f"{get_remaining_life_pct(comp):.0f}%",
            "Action":     cost["action"],
            "Est. Cost":  f"₹{cost['cost']*83:,.0f}",
            "Urgency":    cost["urgency"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

    # ── Export ──
    st.divider()
    e1, e2 = st.columns(2)
    all_export = pd.concat([df.assign(component=c) for c,df in comp_data.items()], ignore_index=True)
    with e1:
        st.download_button("⬇ Sensor Report (CSV)", all_export.to_csv(index=False).encode(),
                           "struxel_sensors.csv","text/csv", use_container_width=True)
    with e2:
        rec_df = pd.DataFrame([{"Priority":r["priority"],"Component":r["component"],
            "Action":r["action"],"Rs":r["Rs"],"Days":r["days_to_failure"],
            "Cost_INR":r["cost"]*83,"Triggered By":r["triggered_by"] or "—"} for r in recommendations])
        st.download_button("⬇ Maintenance Report (CSV)", rec_df.to_csv(index=False).encode(),
                           "struxel_maintenance.csv","text/csv", use_container_width=True)

    if auto_refresh:
        import time; time.sleep(refresh_secs); st.rerun()
