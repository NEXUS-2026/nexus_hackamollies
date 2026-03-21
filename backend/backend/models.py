# models.py — NEXUS ML (IsolationForest + RandomForest) fused with
#             InfraGuard ST-GNN neighbourhood-aware risk propagation
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Sensor thresholds (NEXUS) ─────────────────────────────────────────────────
SAFE_LIMITS = {
    "temperature": {"safe": 75.0,  "warn": 90.0,  "critical": 110.0, "weight": 0.40},
    "pressure":    {"safe": 45.0,  "warn": 55.0,  "critical": 70.0,  "weight": 0.35},
    "vibration":   {"safe": 5.0,   "warn": 7.5,   "critical": 12.0,  "weight": 0.25},
}

# ── InfraGuard Rs weight config ───────────────────────────────────────────────
RS_WEIGHTS = {"w1": 0.40, "w2": 0.35, "w3": 0.25}  # Pf, Centrality, Impact
LAMBDA_HYBRID = 0.65  # local vs neighbourhood in anomaly score


# ═══════════════════════════════════════════════════════════════════════════════
# NEXUS: Anomaly Detection (Isolation Forest)
# ═══════════════════════════════════════════════════════════════════════════════

def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[IsolationForest, StandardScaler]:
    features = df[["temperature", "pressure", "vibration"]].values
    scaler   = StandardScaler()
    model    = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=150)
    model.fit(scaler.fit_transform(features))
    return model, scaler


def detect_anomalies(
    model: IsolationForest,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.Series:
    features = df[["temperature", "pressure", "vibration"]].values
    preds    = model.predict(scaler.transform(features))
    return pd.Series(np.where(preds == -1, "Anomaly", "Normal"), index=df.index, name="anomaly_label")


# ═══════════════════════════════════════════════════════════════════════════════
# NEXUS: Risk Scoring (0–100)
# ═══════════════════════════════════════════════════════════════════════════════

def _sensor_risk(value: float, safe: float, warn: float, critical: float) -> float:
    if value <= safe:
        return (value / safe) * 30
    elif value <= warn:
        return 30 + ((value - safe) / (warn - safe)) * 40
    elif value <= critical:
        return 70 + ((value - warn) / (critical - warn)) * 30
    return 100.0


def calculate_risk_score(row: pd.Series | dict) -> float:
    score = 0.0
    for sensor, cfg in SAFE_LIMITS.items():
        sub    = _sensor_risk(abs(float(row[sensor])), cfg["safe"], cfg["warn"], cfg["critical"])
        score += sub * cfg["weight"]
    return round(min(score, 100.0), 1)


def calculate_risk_scores(df: pd.DataFrame) -> pd.Series:
    return df.apply(calculate_risk_score, axis=1).rename("risk_score")


# ═══════════════════════════════════════════════════════════════════════════════
# NEXUS: Failure Prediction (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════

def train_failure_predictor(
    df: pd.DataFrame,
    risk_threshold: float = 65.0,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, StandardScaler]:
    scores  = calculate_risk_scores(df)
    labels  = (scores >= risk_threshold).astype(int)
    scaler  = StandardScaler()
    features_scaled = scaler.fit_transform(df[["temperature", "pressure", "vibration"]].values)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, class_weight="balanced")
    clf.fit(features_scaled, labels)
    return clf, scaler


def predict_failure(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.DataFrame:
    features_scaled = scaler.transform(df[["temperature", "pressure", "vibration"]].values)
    probs = clf.predict_proba(features_scaled)[:, 1]
    preds = clf.predict(features_scaled)
    return pd.DataFrame({"failure_prob": np.round(probs, 3), "failure_pred": preds}, index=df.index)


def get_system_status(risk_score: float) -> tuple[str, str]:
    if risk_score < 35:   return "✅ Normal",   "#00c853"
    elif risk_score < 65: return "⚠️ At Risk",  "#ff9100"
    else:                 return "🚨 Critical",  "#ff1744"


# ═══════════════════════════════════════════════════════════════════════════════
# InfraGuard: ST-GNN-style neighbourhood-aware anomaly scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_infraguard_rs(
    component_risks: dict[str, float],   # {comp: nexus_risk_0_to_100}
    edges: list[dict],
    component_ages: dict[str, int],
) -> dict[str, dict]:
    """
    Fuses NEXUS 0-100 risk scores with InfraGuard PTDF-weighted
    neighbourhood propagation and Arrhenius age degradation to
    produce the full Rs, Pf, C, I breakdown per component.

    Stage 1: Normalise NEXUS risk → [0,1] → local anomaly score
    Stage 2: PTDF-weighted 1-hop message passing (GAT approximation)
    Stage 3: Hybrid score S_i = λ·local + (1-λ)·Σ PTDF_ij·S_j
    Stage 4: Rs = w1·Pf + w2·C + w3·I
    """
    # Stage 1: local scores
    local = {c: r / 100.0 for c, r in component_risks.items()}

    # Stage 2: neighbourhood scores via PTDF
    nb_scores = {c: 0.0 for c in component_risks}
    for e in edges:
        s, t, ptdf = e["s"], e["t"], e["ptdf"]
        if s in local: nb_scores[t] = nb_scores.get(t, 0) + ptdf * local[s]
        if t in local: nb_scores[s] = nb_scores.get(s, 0) + ptdf * local[t]
    max_nb = max(nb_scores.values()) if nb_scores else 1.0
    if max_nb > 0:
        nb_scores = {c: v / max_nb for c, v in nb_scores.items()}

    # Betweenness-proxy centrality: PTDF column sum
    ptdf_sums = {c: 0.0 for c in component_risks}
    for e in edges:
        ptdf_sums[e["s"]] = ptdf_sums.get(e["s"], 0) + e["ptdf"]
        ptdf_sums[e["t"]] = ptdf_sums.get(e["t"], 0) + e["ptdf"]
    max_ptdf = max(ptdf_sums.values()) if ptdf_sums else 1.0

    result = {}
    for comp, risk_0_100 in component_risks.items():
        ls   = local.get(comp, 0)
        ns   = nb_scores.get(comp, 0)
        S    = LAMBDA_HYBRID * ls + (1 - LAMBDA_HYBRID) * ns

        # Arrhenius age degradation (InfraGuard Phase 1)
        age      = component_ages.get(comp, 5)
        age_risk = min((age / 30) * 0.35, 0.40)

        # P_fail: sigmoid around S=0.5
        pf = 1.0 / (1.0 + math.exp(-5.0 * (S + age_risk - 0.52)))

        # Centrality (normalised PTDF sum)
        c_norm = min(ptdf_sums.get(comp, 0) / max(max_ptdf, 1e-9), 1.0)

        # Impact: approximate from cascade exposure (PTDF to downstream nodes)
        impact = min(sum(e["ptdf"] for e in edges if e["s"] == comp) * 0.8, 1.0)

        # Rs composite
        rs = RS_WEIGHTS["w1"] * pf + RS_WEIGHTS["w2"] * c_norm + RS_WEIGHTS["w3"] * impact

        result[comp] = {
            "Rs":        round(min(rs, 0.99), 4),
            "Pf":        round(pf, 4),
            "C":         round(c_norm, 4),
            "I":         round(impact, 4),
            "S":         round(S, 4),
            "risk_0_100": risk_0_100,
            "anomaly_score": round(S, 4),
        }
    return result
