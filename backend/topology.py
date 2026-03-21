# topology.py — NEXUS component metadata fused with InfraGuard PTDF graph schema
from datetime import date

# ── Real component metadata (from NEXUS) ─────────────────────────────────────
COMPONENT_DETAILS = {
    "Motor": {
        "type": "AC Induction Motor", "manufacturer": "Siemens", "model": "1LE1 Series",
        "install_date": date(2019, 3, 15), "lifespan_years": 10,
        "last_maintenance": date(2025, 11, 20), "power_rating": "75 kW",
        "description": "Primary drive unit powering the pump and compressor.",
        "maint_cost_routine": 2500, "maint_cost_major": 18000, "replacement_cost": 45000,
        "criticality": "High", "x": 180, "y": 160,
        "C_rated": 100.0, "P_rated": 72.0, "R_th": 0.04, "C_th": 4200, "age": 6,
        "reactance_pu": 0.10,
    },
    "Pump": {
        "type": "Centrifugal Pump", "manufacturer": "Grundfos", "model": "CR 95",
        "install_date": date(2020, 7, 10), "lifespan_years": 8,
        "last_maintenance": date(2025, 9, 5), "power_rating": "45 kW",
        "description": "Transfers fluid to the heat exchanger for cooling.",
        "maint_cost_routine": 1800, "maint_cost_major": 12000, "replacement_cost": 32000,
        "criticality": "Medium", "x": 90, "y": 270,
        "C_rated": 100.0, "P_rated": 58.0, "R_th": 0.05, "C_th": 3500, "age": 5,
        "reactance_pu": 0.12,
    },
    "Compressor": {
        "type": "Rotary Screw Compressor", "manufacturer": "Atlas Copco", "model": "GA 55+",
        "install_date": date(2018, 1, 22), "lifespan_years": 12,
        "last_maintenance": date(2025, 6, 18), "power_rating": "55 kW",
        "description": "Compresses gas for the turbine intake stage.",
        "maint_cost_routine": 3200, "maint_cost_major": 22000, "replacement_cost": 65000,
        "criticality": "High", "x": 290, "y": 270,
        "C_rated": 100.0, "P_rated": 81.0, "R_th": 0.045, "C_th": 4800, "age": 8,
        "reactance_pu": 0.09,
    },
    "Heat Exchanger": {
        "type": "Shell & Tube Exchanger", "manufacturer": "Alfa Laval", "model": "M10-BFG",
        "install_date": date(2021, 5, 3), "lifespan_years": 15,
        "last_maintenance": date(2026, 1, 12), "power_rating": "N/A",
        "description": "Removes excess heat from fluid before turbine entry.",
        "maint_cost_routine": 1500, "maint_cost_major": 9000, "replacement_cost": 28000,
        "criticality": "Medium", "x": 90, "y": 380,
        "C_rated": 100.0, "P_rated": 49.0, "R_th": 0.06, "C_th": 5500, "age": 4,
        "reactance_pu": 0.14,
    },
    "Turbine": {
        "type": "Gas Turbine", "manufacturer": "GE Power", "model": "LM2500",
        "install_date": date(2017, 11, 8), "lifespan_years": 15,
        "last_maintenance": date(2025, 12, 1), "power_rating": "30 MW",
        "description": "Converts compressed gas energy into rotational power.",
        "maint_cost_routine": 8500, "maint_cost_major": 55000, "replacement_cost": 250000,
        "criticality": "Critical", "x": 290, "y": 380,
        "C_rated": 100.0, "P_rated": 88.0, "R_th": 0.03, "C_th": 6000, "age": 8,
        "reactance_pu": 0.07,
    },
    "Generator": {
        "type": "Synchronous Generator", "manufacturer": "ABB", "model": "AMG 0900",
        "install_date": date(2017, 11, 8), "lifespan_years": 20,
        "last_maintenance": date(2025, 10, 15), "power_rating": "25 MW",
        "description": "Produces electrical output from turbine rotation.",
        "maint_cost_routine": 6000, "maint_cost_major": 40000, "replacement_cost": 180000,
        "criticality": "Critical", "x": 190, "y": 480,
        "C_rated": 100.0, "P_rated": 76.0, "R_th": 0.035, "C_th": 5800, "age": 8,
        "reactance_pu": 0.08,
    },
}

# ── Directed edges with InfraGuard PTDF physics ──────────────────────────────
EDGES = [
    {"id": "e1", "s": "Motor",        "t": "Pump",          "ptdf": 0.42, "cap": 0.72, "reactance_pu": 0.10},
    {"id": "e2", "s": "Motor",        "t": "Compressor",    "ptdf": 0.38, "cap": 0.81, "reactance_pu": 0.09},
    {"id": "e3", "s": "Pump",         "t": "Heat Exchanger","ptdf": 0.35, "cap": 0.58, "reactance_pu": 0.12},
    {"id": "e4", "s": "Compressor",   "t": "Turbine",       "ptdf": 0.45, "cap": 0.88, "reactance_pu": 0.07},
    {"id": "e5", "s": "Heat Exchanger","t": "Turbine",      "ptdf": 0.28, "cap": 0.49, "reactance_pu": 0.14},
    {"id": "e6", "s": "Turbine",      "t": "Generator",     "ptdf": 0.52, "cap": 0.76, "reactance_pu": 0.08},
]

COMPONENTS = list(COMPONENT_DETAILS.keys())
