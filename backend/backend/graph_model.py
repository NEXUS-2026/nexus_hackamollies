# graph_model.py — NEXUS NetworkX component graph + InfraGuard PTDF cascade physics
import numpy as np
import networkx as nx
from datetime import date
from backend.topology import COMPONENT_DETAILS, EDGES, COMPONENTS
from backend.data_generator import (
    get_component_age_years, get_remaining_life_pct, get_days_since_maintenance
)


# ── Build NetworkX graph ─────────────────────────────────────────────────────

def build_system_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for comp in COMPONENTS:
        info = COMPONENT_DETAILS[comp]
        G.add_node(
            comp,
            risk_score=0.0, Rs=0.0, status="Normal",
            type=info["type"], age_years=get_component_age_years(comp),
            remaining_life_pct=get_remaining_life_pct(comp),
            criticality=info.get("criticality", "Medium"),
        )
    for e in EDGES:
        G.add_edge(e["s"], e["t"], ptdf=e["ptdf"], cap=e["cap"], id=e["id"])
    return G


def assign_risk_to_nodes(graph: nx.DiGraph, component_risks: dict[str, float]) -> nx.DiGraph:
    for node, score in component_risks.items():
        if node in graph.nodes:
            graph.nodes[node]["risk_score"] = score
            graph.nodes[node]["status"] = (
                "Critical" if score >= 65 else "At Risk" if score >= 35 else "Normal"
            )
    return graph


def assign_rs_to_nodes(graph: nx.DiGraph, rs_data: dict[str, dict]) -> nx.DiGraph:
    for node, data in rs_data.items():
        if node in graph.nodes:
            graph.nodes[node].update(data)
    return graph


def get_node_colors(graph: nx.DiGraph) -> list[str]:
    colors = []
    for node in graph.nodes:
        rs = graph.nodes[node].get("Rs", 0)
        if rs >= 0.70:   colors.append("#ff1744")
        elif rs >= 0.50: colors.append("#ff9100")
        elif rs >= 0.35: colors.append("#f59e0b")
        else:            colors.append("#00c853")
    return colors


def get_graph_layout(graph: nx.DiGraph) -> dict:
    # Use fixed positions from topology for deterministic layout
    positions = {}
    for node in graph.nodes:
        info = COMPONENT_DETAILS.get(node, {})
        positions[node] = (info.get("x", 200), -info.get("y", 200))  # flip y for plotly
    return positions


# ── InfraGuard PTDF matrix ───────────────────────────────────────────────────

def build_ptdf(node_list: list[str], edge_list: list[dict]) -> dict:
    """
    Compute Power Transfer Distribution Factor matrix via DC power flow.
    PTDF[j][i] = fraction of node i's lost load redistributed to node j.
    """
    N   = len(node_list)
    idx = {n: i for i, n in enumerate(node_list)}
    B   = np.zeros((N, N))
    for e in edge_list:
        if e["s"] not in idx or e["t"] not in idx:
            continue
        i, j = idx[e["s"]], idx[e["t"]]
        b_ij = 1.0 / max(e.get("reactance_pu", 0.1), 1e-6)
        B[i,i] += b_ij; B[j,j] += b_ij; B[i,j] -= b_ij; B[j,i] -= b_ij
    B_red = B[1:, 1:]
    B_inv = np.zeros((N, N))
    B_inv[1:, 1:] = np.linalg.pinv(B_red)
    ptdf = {}
    for j, jid in enumerate(node_list):
        ptdf[jid] = {}
        for i, iid in enumerate(node_list):
            ptdf[jid][iid] = float(B_inv[j, i])
    return ptdf


# ── Thermal model (InfraGuard RC circuit) ───────────────────────────────────

def update_temperature(T: float, P_pct: float, T_amb=25.0, R_th=0.05, C_th=5000.0, dt=72.0) -> float:
    P_loss = (P_pct / 100) ** 2 * 15.0
    return T + (dt / C_th) * (P_loss - (T - T_amb) / R_th)


def capacity_degradation(T_hotspot: float, age: int) -> float:
    """Arrhenius: every 6°C above 98°C halves insulation life (IEC 60076-7)."""
    delta_T = max(T_hotspot - 98.0, 0)
    aging   = 2.0 ** (delta_T / 6.0)
    age_fac = 1.0 - min(age / 60.0, 0.45)
    return max(1.0 / (1.0 + 0.006 * (aging - 1.0)) * age_fac, 0.10)


# ── InfraGuard deterministic cascade simulation ──────────────────────────────

def simulate_failure_propagation(
    graph:            nx.DiGraph,
    failed_node:      str,
    component_loads:  dict[str, float] | None = None,
    stressor_type:    str = "load_surge",
    delta_load:       float = 0.5,
    delta_temp:       float = 20.0,
) -> tuple[nx.DiGraph, list[dict]]:
    """
    InfraGuard PTDF redistribution cascade (replaces NEXUS BFS propagation).
    Preserves load conservation. Returns updated graph + cascade log.
    """
    import copy
    node_list = list(graph.nodes)
    edge_list = [{"s": u, "t": v, **graph.edges[u,v]} for u,v in graph.edges]

    # Node states
    states = {}
    for n in node_list:
        info = COMPONENT_DETAILS.get(n, {})
        load = (component_loads or {}).get(n, info.get("P_rated", 60.0))
        temp = 75.0 + info.get("age", 5) * 0.8
        states[n] = {
            "P_current":  load,
            "C_current":  100.0 * capacity_degradation(temp, info.get("age", 5)),
            "T_hotspot":  temp,
            "status":     "ok",
            "step":       None,
            "fail_reason": None,
            "R_th":       info.get("R_th", 0.05),
            "C_th":       info.get("C_th", 5000),
            "age":        info.get("age", 5),
        }

    # Apply stressor to seed node
    s = states[failed_node]
    if stressor_type == "thermal":
        s["T_hotspot"]  += delta_temp
        s["C_current"]  *= capacity_degradation(s["T_hotspot"], s["age"])
    elif stressor_type == "load_surge":
        s["P_current"]  *= (1.0 + delta_load)
        s["T_hotspot"]   = update_temperature(s["T_hotspot"], s["P_current"], R_th=s["R_th"], C_th=s["C_th"])
    elif stressor_type == "compound":
        s["T_hotspot"]  += delta_temp
        s["C_current"]  *= capacity_degradation(s["T_hotspot"], s["age"])
        s["P_current"]  *= (1.0 + delta_load * 0.6)

    cascade_log = []
    failed_ids  = set()

    for step in range(10):
        newly_failed = []
        for n, st in states.items():
            if st["status"] != "ok":
                continue
            h = (st["C_current"] - st["P_current"]) / max(st["C_current"], 1)
            if h < 0:
                st["status"] = "Critical"; st["step"] = step
                st["fail_reason"] = f"overload H={h:.3f}"
                newly_failed.append(n)
            elif st["T_hotspot"] > 135:
                st["status"] = "Critical"; st["step"] = step
                st["fail_reason"] = f"thermal T={st['T_hotspot']:.1f}°C"
                newly_failed.append(n)

        if not newly_failed:
            break
        failed_ids.update(newly_failed)

        surviving_nodes = [n for n in node_list if states[n]["status"] == "ok"]
        surviving_edges = [e for e in edge_list if e["s"] not in failed_ids and e["t"] not in failed_ids]
        ptdf = build_ptdf(surviving_nodes, surviving_edges)

        step_log = {"step": step, "newly_failed": newly_failed, "redistributions": {}}
        for fn in newly_failed:
            P_lost = states[fn]["P_current"]
            step_log["redistributions"][fn] = {}
            for n in surviving_nodes:
                dp = ptdf.get(n, {}).get(fn, 0) * P_lost
                if abs(dp) > 0.01:
                    states[n]["P_current"] = max(states[n]["P_current"] + dp, 0)
                    states[n]["T_hotspot"] = update_temperature(
                        states[n]["T_hotspot"], states[n]["P_current"],
                        R_th=states[n]["R_th"], C_th=states[n]["C_th"])
                    step_log["redistributions"][fn][n] = round(dp, 2)

                    # Promote to "At Risk" if load spikes
                    if states[n]["P_current"] > states[n]["C_current"] * 0.82:
                        if states[n]["status"] == "ok":
                            states[n]["status"] = "At Risk"

        cascade_log.append(step_log)

    # Write results back to graph
    for n in graph.nodes:
        st = states[n]
        graph.nodes[n]["status"]      = st["status"]
        graph.nodes[n]["risk_score"]  = min(st["P_current"], 100.0)
        graph.nodes[n]["load_pct"]    = round(st["P_current"], 1)
        graph.nodes[n]["T_hotspot"]   = round(st["T_hotspot"], 1)
        graph.nodes[n]["cascade_step"]= st["step"]
        graph.nodes[n]["fail_reason"] = st["fail_reason"]
        if st["status"] == "Critical":
            graph.nodes[n]["Rs"] = min(graph.nodes[n].get("Rs", 0.5) + 0.3, 0.99)
        elif st["status"] == "At Risk":
            graph.nodes[n]["Rs"] = min(graph.nodes[n].get("Rs", 0.3) + 0.15, 0.85)

    return graph, cascade_log


# ── NEXUS helpers ────────────────────────────────────────────────────────────

def build_component_summary() -> list[dict]:
    rows = []
    for name, info in COMPONENT_DETAILS.items():
        age          = get_component_age_years(name)
        remaining    = get_remaining_life_pct(name)
        maint_days   = get_days_since_maintenance(name)
        if remaining > 50:   aging_label, aging_color = "Good",               "#00c853"
        elif remaining > 25: aging_label, aging_color = "Aging",              "#ff9100"
        elif remaining > 0:  aging_label, aging_color = "Near End-of-Life",   "#ff5252"
        else:                aging_label, aging_color = "Overdue",            "#d50000"
        rows.append({
            "Component": name, "Type": info["type"],
            "Manufacturer": info["manufacturer"], "Model": info["model"],
            "Power Rating": info["power_rating"],
            "Install Date": info["install_date"].strftime("%d %b %Y"),
            "Age (Years)": age, "Lifespan (Years)": info["lifespan_years"],
            "Remaining Life %": remaining,
            "Last Maintenance": info["last_maintenance"].strftime("%d %b %Y"),
            "Days Since Maint.": maint_days,
            "Aging Status": aging_label, "Criticality": info.get("criticality", "Medium"),
            "Description": info["description"], "_color": aging_color,
        })
    return rows


def estimate_days_to_failure(component: str, risk_score: float) -> int:
    remaining_pct  = get_remaining_life_pct(component)
    lifespan_days  = COMPONENT_DETAILS.get(component, {}).get("lifespan_years", 10) * 365
    base_days      = max(0, remaining_pct / 100 * lifespan_days)
    factor         = 0.15 if risk_score >= 65 else 0.4 if risk_score >= 45 else 0.65 if risk_score >= 35 else 1.0
    return max(1, int(base_days * factor))


def estimate_maintenance_cost(component: str, risk_score: float) -> dict:
    info      = COMPONENT_DETAILS.get(component, {})
    routine   = info.get("maint_cost_routine", 2000)
    major     = info.get("maint_cost_major", 15000)
    replace   = info.get("replacement_cost", 50000)
    remaining = get_remaining_life_pct(component)
    if risk_score >= 65 or remaining < 10:
        return {"action": "Major Overhaul / Replace", "cost": replace if remaining < 5 else major, "urgency": "Immediate", "urgency_color": "#ff1744"}
    elif risk_score >= 40 or remaining < 25:
        return {"action": "Major Maintenance",  "cost": major,   "urgency": "Within 2 weeks", "urgency_color": "#ff9100"}
    elif risk_score >= 25 or remaining < 50:
        return {"action": "Routine Service",    "cost": routine, "urgency": "Scheduled",      "urgency_color": "#818cf8"}
    return {"action": "No action needed", "cost": 0, "urgency": "None", "urgency_color": "#00c853"}


def get_system_health_score(component_risks: dict[str, float]) -> float:
    weights = {"Critical": 3.0, "High": 2.0, "Medium": 1.0}
    total_w = total_h = 0.0
    for comp, risk in component_risks.items():
        w       = weights.get(COMPONENT_DETAILS.get(comp, {}).get("criticality", "Medium"), 1.0)
        total_w += w; total_h += max(0, 100 - risk) * w
    return round(total_h / total_w, 1) if total_w > 0 else 100.0


def generate_causal_maintenance_recommendations(
    component_risks: dict[str, float],
    rs_data: dict[str, dict],
    graph: nx.DiGraph,
) -> list[dict]:
    """
    InfraGuard causal recommender: each recommendation includes the
    upstream node responsible for the stress (via PTDF edge ancestry).
    """
    recommendations = []
    for comp in sorted(component_risks, key=lambda c: -rs_data.get(c, {}).get("Rs", 0)):
        rs   = rs_data.get(comp, {}).get("Rs", 0)
        risk = component_risks.get(comp, 0)
        if rs < 0.30 and risk < 35:
            continue

        # Find upstream neighbour with highest PTDF·Rs product
        triggered_by = None
        best_score   = 0
        for pred in graph.predecessors(comp):
            edge_data  = graph.edges.get((pred, comp), {})
            ptdf       = edge_data.get("ptdf", 0)
            pred_rs    = rs_data.get(pred, {}).get("Rs", 0)
            if ptdf * pred_rs > best_score:
                best_score   = ptdf * pred_rs
                triggered_by = pred

        # Priority
        if rs >= 0.70 or risk >= 65: priority, horizon = 1, "24 hours"
        elif rs >= 0.50 or risk >= 45: priority, horizon = 2, "7 days"
        else: priority, horizon = 3, "Next scheduled"

        cost_info = estimate_maintenance_cost(comp, risk)
        days_left = estimate_days_to_failure(comp, risk)

        # Causal chain
        chain = []
        if triggered_by:
            src_risk = component_risks.get(triggered_by, 0)
            edge_ptdf = graph.edges.get((triggered_by, comp), {}).get("ptdf", 0)
            chain.append(f"'{triggered_by}' operating at risk score {src_risk:.0f}/100 "
                         f"with PTDF coupling {edge_ptdf:.2f} to '{comp}'.")
            chain.append(f"Load redistribution from '{triggered_by}' is increasing "
                         f"stress on '{comp}' via the dependency edge.")
        remaining = get_remaining_life_pct(comp)
        chain.append(f"'{comp}' has {remaining}% remaining useful life, "
                     f"risk score {risk:.0f}/100, Rs={rs:.3f}.")
        chain.append(f"Estimated {days_left} days to failure at current degradation rate. "
                     f"Recommended: {cost_info['action']} (${cost_info['cost']:,}).")

        recommendations.append({
            "rank": len(recommendations) + 1,
            "priority": priority,
            "horizon": horizon,
            "component": comp,
            "action": cost_info["action"],
            "cost": cost_info["cost"],
            "urgency": cost_info["urgency"],
            "urgency_color": cost_info["urgency_color"],
            "risk_score": risk,
            "Rs": rs,
            "days_to_failure": days_left,
            "triggered_by": triggered_by,
            "causal_chain": chain,
            "remaining_life_pct": remaining,
        })
    return recommendations
