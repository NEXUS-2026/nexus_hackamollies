# ⬡ NexusGuard — Predictive Maintenance Platform
### NEXUS × InfraGuard Fusion | DS-PS2 Submission

---

## 🚀 Two Ways to Run

### Option 1 — Standalone HTML (Zero install)
Just open `dashboard.html` in any browser.
Full interactive dashboard with **light/dark glassmorphism toggle**.

### Option 2 — Streamlit App (Full Python backend)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at → **http://localhost:8501**

---

## 📁 Project Structure

```
nexusguard/
│
├── dashboard.html          ← Standalone interactive dashboard (open in browser)
├── app.py                  ← Streamlit entry point
├── requirements.txt        ← Python dependencies
│
├── backend/
│   ├── topology.py         ← Real component metadata + PTDF edge schema
│   ├── data_generator.py   ← Sensor simulation (Weibull, LogNormal, demand cycle)
│   ├── models.py           ← IsolationForest + RandomForest + InfraGuard Rs scoring
│   └── graph_model.py      ← NetworkX graph + PTDF cascade engine + recommender
│
└── frontend/
    ├── app.py              ← Streamlit UI pages
    └── visualizations.py   ← Plotly chart builders
```

---

## ✨ Features

| Feature | Source | Description |
|---|---|---|
| Sensor simulation | NEXUS | Temperature, pressure, vibration with anomaly injection |
| Anomaly detection | NEXUS | Isolation Forest (sklearn) per component |
| Failure prediction | NEXUS | Random Forest classifier |
| Rs risk scoring | InfraGuard | Rs = w1·Pf + w2·C + w3·I (neighbourhood-aware) |
| PTDF cascade | InfraGuard | Deterministic load redistribution (physics-based) |
| Causal maintenance | InfraGuard | Chain-of-failure recommendations with upstream trigger |
| Real components | NEXUS | Siemens, GE, ABB, Atlas Copco — real metadata |
| Light/Dark toggle | Merged | Glassmorphism aurora — both themes |
| Network graph | Merged | Rs-colored nodes, PTDF-weighted edges |
| Anomaly charts | Merged | Live time-series with danger zone overlays |
| Resilience scorecard | Merged | Economic impact, downtime prevented |
| CSV export | NEXUS | One-click sensor + maintenance report download |

---

## 🎨 Dashboard Themes

The standalone `dashboard.html` includes both themes with a toggle in the header:

- **🌙 Dark Glassmorphism** — Deep violet/teal aurora, smoked glass panels, neon data colors
- **☀️ Light Glassmorphism** — Pastel lavender/mint aurora, bright frosted panels, jewel-tone colors

---

## 🧠 Phase Coverage (DS-PS2)

| Requirement | Phase | Implementation |
|---|---|---|
| Real-time IoT processing | Phase 2 | Sensor simulation → IsoForest → Rs scoring |
| Graph-based learning | Phase 2 | PTDF-weighted neighbourhood anomaly propagation |
| Cascade simulation | Phase 3 | Deterministic PTDF redistribution engine |
| Dynamic risk scoring | Phase 3 | Rs = 0.40·Pf + 0.35·C + 0.25·I |
| Preventive maintenance | Phase 3 | Causal chain recommender |
| Real-time dashboard | Phase 4 | Interactive HTML + Streamlit |
| Anomaly trend graphs | Phase 4 | Vibration + Temperature + Pressure |
| Failure heatmaps | Phase 4 | Component risk heatmap |
| Network graphs | Phase 4 | Rs-colored topology graph |
| Economic metrics | Phase 4 | Resilience scorecard |

---

## 🛠️ Tech Stack

- **Frontend**: Pure HTML/CSS/JS (standalone) + Streamlit (Python)
- **ML**: scikit-learn (IsolationForest, RandomForest)
- **Physics**: DC Power Flow (PTDF), Arrhenius thermal model, RC circuit
- **Visualization**: Plotly, HTML5 Canvas
- **Graph**: NetworkX + custom PTDF solver
- **Fonts**: Sora + JetBrains Mono
