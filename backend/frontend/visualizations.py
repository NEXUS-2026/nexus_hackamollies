# visualizations.py — NEXUS Plotly charts enhanced with InfraGuard Rs/PTDF overlays
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx

COLORS = {
    "bg": "#0e1117", "card": "#1a1d23", "text": "#e0e0e0",
    "primary": "#6366f1", "green": "#00c853", "orange": "#ff9100",
    "red": "#ff1744", "line": "#818cf8", "anomaly": "#ff1744",
    "grid": "#2a2d35", "teal": "#00d4aa", "amber": "#f5a623",
}
LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="IBM Plex Mono, monospace"),
    margin=dict(l=44, r=20, t=44, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
)

def _base(**overrides):
    return {**LAYOUT_DEFAULTS, **overrides}


# ── Sensor time-series with anomaly highlights ───────────────────────────────

def plot_sensor_timeseries(df: pd.DataFrame, sensor: str, component: str = "") -> go.Figure:
    normal  = df[df["anomaly_label"] == "Normal"]  if "anomaly_label" in df else df
    anomaly = df[df["anomaly_label"] == "Anomaly"] if "anomaly_label" in df else df.iloc[0:0]

    # Danger zone thresholds (InfraGuard Phase 1 values)
    danger_levels = {"temperature": 90.0, "pressure": 55.0, "vibration": 7.5}
    danger_val    = danger_levels.get(sensor, df[sensor].max() * 0.85)

    fig = go.Figure()
    fig.add_hrect(y0=danger_val, y1=df[sensor].max() * 1.1,
                  fillcolor="rgba(255,23,68,0.08)", line_width=0,
                  annotation_text="DANGER ZONE", annotation_position="top left",
                  annotation_font_color=COLORS["red"])
    fig.add_hline(y=danger_val, line_dash="dash", line_color=COLORS["orange"],
                  line_width=1.2, annotation_text=f"Threshold {danger_val}",
                  annotation_font_color=COLORS["orange"])
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal[sensor], mode="lines",
        name="Normal", line=dict(color=COLORS["teal"], width=1.6),
    ))
    if not anomaly.empty:
        fig.add_trace(go.Scatter(
            x=anomaly["timestamp"], y=anomaly[sensor], mode="markers",
            name="Anomaly", marker=dict(color=COLORS["red"], size=7, symbol="x"),
        ))
    units = {"temperature": "°C", "pressure": "bar", "vibration": "m/s²"}
    fig.update_layout(**_base(title=f"{component} — {sensor.title()} ({units.get(sensor,'')})"))
    return fig


# ── Risk heatmap ─────────────────────────────────────────────────────────────

def plot_risk_heatmap(component_risks: dict[str, float]) -> go.Figure:
    comps  = list(component_risks.keys())
    scores = list(component_risks.values())
    fig = go.Figure(go.Bar(
        x=scores, y=comps, orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0,"#00c853"],[0.35,"#f59e0b"],[0.65,"#ff9100"],[1,"#ff1744"]],
            cmin=0, cmax=100,
            colorbar=dict(title="Risk", tickfont=dict(color=COLORS["text"])),
        ),
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(**_base(title="Component Risk Heatmap (0–100)", xaxis=dict(range=[0,110], gridcolor=COLORS["grid"])))
    return fig


# ── Rs bar chart (InfraGuard) ─────────────────────────────────────────────────

def plot_rs_breakdown(rs_data: dict[str, dict]) -> go.Figure:
    comps = list(rs_data.keys())
    pf    = [rs_data[c]["Pf"] for c in comps]
    c_    = [rs_data[c]["C"]  for c in comps]
    imp   = [rs_data[c]["I"]  for c in comps]
    rs    = [rs_data[c]["Rs"] for c in comps]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="P_fail (w=0.40)", x=comps, y=pf,    marker_color="#ff5252"))
    fig.add_trace(go.Bar(name="Centrality (w=0.35)", x=comps, y=c_, marker_color="#6366f1"))
    fig.add_trace(go.Bar(name="Impact (w=0.25)", x=comps, y=imp,   marker_color="#f59e0b"))
    fig.add_trace(go.Scatter(name="Rs (composite)", x=comps, y=rs,
                             mode="lines+markers", line=dict(color=COLORS["teal"], width=2.5, dash="dot"),
                             marker=dict(size=8)))
    fig.update_layout(**_base(barmode="stack",
                              title="InfraGuard Rs Decomposition: P_fail · Centrality · Impact",
                              yaxis=dict(range=[0, 1.05], gridcolor=COLORS["grid"])))
    return fig


# ── Network graph with Rs + PTDF edge thickness ──────────────────────────────

def plot_network_graph(graph: nx.DiGraph, rs_data: dict[str, dict] | None = None) -> go.Figure:
    from backend.graph_model import get_graph_layout
    pos = get_graph_layout(graph)

    edge_traces = []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ptdf    = data.get("ptdf", 0.2)
        cap     = data.get("cap", 0.5)
        color   = COLORS["red"] if cap > 0.80 else COLORS["orange"] if cap > 0.65 else "#2a3f6a"
        width   = 1.5 + ptdf * 5   # InfraGuard: PTDF drives edge thickness
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none", showlegend=False,
        ))
        # PTDF label
        edge_traces.append(go.Scatter(
            x=[(x0+x1)/2], y=[(y0+y1)/2],
            mode="text", text=[f"PTDF={ptdf:.2f}"],
            textfont=dict(size=8, color="#6b7a99"),
            hoverinfo="none", showlegend=False,
        ))

    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    for node in graph.nodes:
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        status  = graph.nodes[node].get("status", "Normal")
        rs      = (rs_data or {}).get(node, {}).get("Rs", 0)
        risk    = graph.nodes[node].get("risk_score", 0)
        color   = "#ff1744" if status == "Critical" else "#ff9100" if status == "At Risk" else "#00c853"
        node_colors.append(color)
        node_sizes.append(22 + rs * 20)
        node_text.append(
            f"<b>{node}</b><br>Rs={rs:.3f}<br>Risk={risk:.0f}/100<br>Status={status}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(graph.nodes),
        textposition="top center",
        textfont=dict(size=10, color=COLORS["text"]),
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=2, color="#0e1117")),
        hovertext=node_text, hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_base(
        title="System Dependency Graph — Node size ∝ Rs · Edge width ∝ PTDF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=440,
    ))
    return fig


# ── Risk gauge (NEXUS) ───────────────────────────────────────────────────────

def plot_risk_gauge(risk_score: float, title: str = "System Risk") -> go.Figure:
    color = "#ff1744" if risk_score >= 65 else "#ff9100" if risk_score >= 35 else "#00c853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": 35, "increasing": {"color": "#ff1744"}, "decreasing": {"color": "#00c853"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text"]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#0a2818"},
                {"range": [35, 65], "color": "#2a1800"},
                {"range": [65, 100],"color": "#2a0a10"},
            ],
            "threshold": {"line": {"color": "#ff1744", "width": 3}, "value": 65},
        },
        title={"text": title, "font": {"color": COLORS["text"]}},
        number={"font": {"color": color}},
    ))
    fig.update_layout(**_base(height=280, margin=dict(l=20, r=20, t=60, b=20)))
    return fig


# ── Rs gauge (InfraGuard) ────────────────────────────────────────────────────

def plot_rs_gauge(rs: float, component: str) -> go.Figure:
    color = "#ff1744" if rs >= 0.70 else "#ff9100" if rs >= 0.50 else "#f59e0b" if rs >= 0.35 else "#00c853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(rs, 3),
        gauge={
            "axis": {"range": [0, 1], "tickcolor": COLORS["text"]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 0.35], "color": "#0a2818"},
                {"range": [0.35, 0.55], "color": "#1a1a00"},
                {"range": [0.55, 0.75], "color": "#2a1800"},
                {"range": [0.75, 1.0],  "color": "#2a0a10"},
            ],
        },
        title={"text": f"Rs — {component}", "font": {"color": COLORS["text"]}},
        number={"font": {"color": color}, "valueformat": ".3f"},
    ))
    fig.update_layout(**_base(height=260, margin=dict(l=20, r=20, t=60, b=20)))
    return fig


# ── Aging chart (NEXUS) ──────────────────────────────────────────────────────

def plot_aging_chart(summary: list[dict]) -> go.Figure:
    df = pd.DataFrame(summary)
    colors = [row["_color"] for _, row in df.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Component"], y=df["Remaining Life %"],
        marker_color=colors,
        text=[f"{v:.0f}%" for v in df["Remaining Life %"]],
        textposition="outside", textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(**_base(title="Remaining Useful Life (%) by Component",
                              yaxis=dict(range=[0, 115], gridcolor=COLORS["grid"])))
    return fig


# ── PTDF cascade impact bar (InfraGuard) ─────────────────────────────────────

def plot_cascade_impact(cascade_log: list[dict], all_components: list[str]) -> go.Figure:
    impact = {c: 0 for c in all_components}
    for entry in cascade_log:
        for failed, redist in entry.get("redistributions", {}).items():
            for comp, dp in redist.items():
                impact[comp] = impact.get(comp, 0) + abs(dp)

    comps  = list(impact.keys())
    values = list(impact.values())
    colors = ["#ff1744" if v > 20 else "#ff9100" if v > 8 else "#6366f1" for v in values]

    fig = go.Figure(go.Bar(
        x=comps, y=values, marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(**_base(title="PTDF Load Redistributed to Each Component (MW-equivalent)"))
    return fig


# ── Maintenance timeline ─────────────────────────────────────────────────────

def plot_maintenance_timeline(recommendations: list[dict]) -> go.Figure:
    if not recommendations:
        return go.Figure()
    df = pd.DataFrame(recommendations)
    color_map = {1: "#ff1744", 2: "#ff9100", 3: "#6366f1"}
    colors    = [color_map.get(p, "#6366f1") for p in df["priority"]]
    fig = go.Figure(go.Bar(
        x=df["component"], y=df["days_to_failure"],
        marker_color=colors,
        text=[f"P{p} · {h}" for p, h in zip(df["priority"], df["horizon"])],
        textposition="outside", textfont=dict(color=COLORS["text"]),
        customdata=df["action"],
        hovertemplate="<b>%{x}</b><br>Days to failure: %{y}<br>Action: %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=30, line_dash="dash", line_color="#ff1744",
                  annotation_text="30-day critical window", annotation_font_color="#ff1744")
    fig.update_layout(**_base(title="Estimated Days to Failure + Maintenance Priority",
                              yaxis=dict(gridcolor=COLORS["grid"])))
    return fig
