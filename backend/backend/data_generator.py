# data_generator.py — NEXUS sensor simulation + InfraGuard Weibull/LogNormal stressors
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

BASELINES = {
    "temperature": {"mean": 65.0, "std": 5.0,  "spike_min": 95.0,  "spike_max": 120.0},
    "pressure":    {"mean": 35.0, "std": 4.0,  "spike_min": 60.0,  "spike_max": 80.0},
    "vibration":   {"mean": 3.5,  "std": 1.0,  "spike_min": 8.0,   "spike_max": 15.0},
}

# ── Per-component sensor offsets (realistic variation) ───────────────────────
COMPONENT_OFFSETS = {
    "Motor":         {"temperature": 8.0,  "pressure":  2.0, "vibration": 0.5},
    "Pump":          {"temperature": 3.0,  "pressure":  5.0, "vibration": 1.2},
    "Compressor":    {"temperature": 12.0, "pressure":  8.0, "vibration": 1.8},
    "Heat Exchanger":{"temperature": -5.0, "pressure":  3.0, "vibration": 0.3},
    "Turbine":       {"temperature": 18.0, "pressure": 10.0, "vibration": 2.1},
    "Generator":     {"temperature": 10.0, "pressure":  1.0, "vibration": 0.8},
}


def generate_sensor_data(
    n_points: int = 500,
    anomaly_ratio: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """Standard NEXUS-style sensor DataFrame."""
    rng = np.random.default_rng(seed)
    start = datetime.now() - timedelta(seconds=10 * n_points)
    timestamps = [start + timedelta(seconds=10 * i) for i in range(n_points)]

    temperature = rng.normal(BASELINES["temperature"]["mean"], BASELINES["temperature"]["std"], n_points)
    pressure    = rng.normal(BASELINES["pressure"]["mean"],    BASELINES["pressure"]["std"],    n_points)
    vibration   = rng.normal(BASELINES["vibration"]["mean"],   BASELINES["vibration"]["std"],   n_points)

    n_anomalies = max(1, int(n_points * anomaly_ratio))
    anomaly_idx = rng.choice(n_points, size=n_anomalies, replace=False)
    is_anomaly  = np.zeros(n_points, dtype=bool)
    is_anomaly[anomaly_idx] = True

    for sensor, arr in [("temperature", temperature), ("pressure", pressure), ("vibration", vibration)]:
        cfg = BASELINES[sensor]
        arr[anomaly_idx] = rng.uniform(cfg["spike_min"], cfg["spike_max"], size=n_anomalies)

    return pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": np.round(temperature, 2),
        "pressure":    np.round(pressure, 2),
        "vibration":   np.round(vibration, 2),
        "is_anomaly":  is_anomaly,
    })


def generate_component_data(
    components: list[str],
    n_points: int = 200,
    anomaly_ratio: float = 0.05,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate per-component sensor data with realistic offsets.
    Mirrors NEXUS generate_component_data but with InfraGuard-style
    age-driven baseline drift.
    """
    from backend.topology import COMPONENT_DETAILS
    rng = np.random.default_rng(seed)
    result = {}
    for comp in components:
        info    = COMPONENT_DETAILS.get(comp, {})
        age     = info.get("age", 5)
        offsets = COMPONENT_OFFSETS.get(comp, {"temperature": 0, "pressure": 0, "vibration": 0})
        comp_seed = int(rng.integers(0, 2**31))
        df = generate_sensor_data(n_points, anomaly_ratio, seed=comp_seed)
        # Age-based drift: older components run hotter and vibrate more
        df["temperature"] += offsets["temperature"] + age * 0.4
        df["pressure"]    += offsets["pressure"]    + age * 0.15
        df["vibration"]   += offsets["vibration"]   + age * 0.06
        df["component"]    = comp
        result[comp] = df
    return result


def generate_realtime_point(
    component: str | None = None,
    rng: np.random.Generator | None = None,
    force_anomaly: bool = False,
    tick: int = 0,
) -> dict:
    """
    Single live reading — NEXUS style but with InfraGuard demand-cycle physics.
    """
    from backend.topology import COMPONENT_DETAILS
    if rng is None:
        rng = np.random.default_rng()

    is_anomaly = force_anomaly or (rng.random() < 0.06)
    offsets    = COMPONENT_OFFSETS.get(component, {"temperature": 0, "pressure": 0, "vibration": 0})
    age        = COMPONENT_DETAILS.get(component, {}).get("age", 5) if component else 5

    # InfraGuard-style demand sinusoid
    demand_cycle = np.sin(tick / 60 + (hash(component or "") % 100) * 0.1) * 6

    if is_anomaly:
        # InfraGuard Weibull mechanical + LogNormal thermal spikes
        temp = float(rng.lognormal(mean=np.log(BASELINES["temperature"]["spike_min"]), sigma=0.3))
        pres = float(rng.uniform(BASELINES["pressure"]["spike_min"], BASELINES["pressure"]["spike_max"]))
        vib  = float(rng.weibull(2.0) * BASELINES["vibration"]["spike_max"] * 0.7)
    else:
        temp = float(rng.normal(BASELINES["temperature"]["mean"] + offsets["temperature"] + age * 0.4 + demand_cycle, BASELINES["temperature"]["std"]))
        pres = float(rng.normal(BASELINES["pressure"]["mean"]    + offsets["pressure"]    + age * 0.15,                BASELINES["pressure"]["std"]))
        vib  = float(abs(rng.normal(BASELINES["vibration"]["mean"] + offsets["vibration"] + age * 0.06,               BASELINES["vibration"]["std"])))
        # Occasional kurtosis event (InfraGuard bearing impact model)
        if rng.random() < 0.02:
            vib += float(rng.uniform(2.0, 5.0))

    return {
        "timestamp":   datetime.now(),
        "component":   component,
        "temperature": round(temp, 2),
        "pressure":    round(pres, 2),
        "vibration":   round(vib, 3),
        "is_anomaly":  is_anomaly,
    }


def get_component_age_years(component: str, reference_date: date | None = None) -> float:
    from backend.topology import COMPONENT_DETAILS
    from datetime import date as d
    if reference_date is None:
        reference_date = d.today()
    install = COMPONENT_DETAILS.get(component, {}).get("install_date", reference_date)
    return round((reference_date - install).days / 365.25, 1)


def get_remaining_life_pct(component: str) -> float:
    from backend.topology import COMPONENT_DETAILS
    age      = get_component_age_years(component)
    lifespan = COMPONENT_DETAILS.get(component, {}).get("lifespan_years", 10)
    return max(0.0, round((1 - age / lifespan) * 100, 1))


def get_days_since_maintenance(component: str) -> int:
    from backend.topology import COMPONENT_DETAILS
    from datetime import date as d
    last = COMPONENT_DETAILS.get(component, {}).get("last_maintenance", d.today())
    return (d.today() - last).days
