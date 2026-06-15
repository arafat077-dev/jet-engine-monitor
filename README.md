[README.md](https://github.com/user-attachments/files/28969555/README.md)
# JetGuard AI — Jet Engine Health Monitoring System

A dual-model machine learning system that predicts jet engine failure before it happens, using anomaly detection and Remaining Useful Life (RUL) regression on the NASA C-MAPSS dataset.

**Live app:** deployed on Streamlit Community Cloud
**Course:** Basic Design (공학설계기초) · Sejong University
**Instructor:** Prof. Abolghasem Sadeghi-Niaraki
**Team:** Group 2

---

## What It Does

JetGuard AI monitors jet engine sensor data and answers two questions at once:

1. **Is this engine behaving abnormally right now?** — handled by an Isolation Forest anomaly detector
2. **Exactly how many cycles until failure?** — handled by a Gradient Boosting RUL regressor

The two models run **in parallel**, so a developing failure is only missed if *both* models miss it. Their predictions are combined into a four-level health status (HEALTHY / MONITOR / WARNING / CRITICAL) with a maintenance recommendation.

---

## Key Results

| Metric | Result | Requirement | Status |
|---|---|---|---|
| Detection lead time | 71.7 cycles | ≥ 15 | Pass |
| RUL prediction MAE | 15.11 cycles | ≤ 20 | Pass |
| Near-failure MAE | 8.08 cycles | ≤ 20 | Pass |
| False positive rate | 4.54% | < 5% | Pass |
| Danger-zone coverage | 60.2% | ≥ 90% | Partial — see note |
| Anomaly rate | 14.0% | ≤ 15% | Pass |
| F1-score (danger zone) | 0.62 | — | Reported |
| Combined danger detection (parallel) | ~88% | — | Dual-model |

**Note on coverage:** unsupervised anomaly detection plateaus at ~60% danger-zone coverage — a documented algorithmic limit, not a tuning failure. The system addresses this with a second model: the Gradient Boosting regressor predicts RUL directly and catches the smooth-degradation engines the anomaly detector misses, raising combined danger detection to roughly 88%.

---

## Dual-Model Architecture

```
                    ┌─────────────────────────┐
   Sensor data ───► │  Preprocessing (scale)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Feature Engineering     │
                    │  45 features:            │
                    │  base + rolling + lag    │
                    │  + delta                 │
                    └──────┬───────────┬───────┘
                           │           │
              ┌────────────▼──┐   ┌────▼─────────────┐
              │ Isolation     │   │ Gradient Boosting│
              │ Forest V3     │   │ RUL Regressor    │
              │ (anomaly)     │   │ (RUL prediction) │
              └────────────┬──┘   └────┬─────────────┘
                           │           │
                    ┌──────▼───────────▼──────┐
                    │  Health Status Logic     │
                    │  HEALTHY / MONITOR /     │
                    │  WARNING / CRITICAL      │
                    └──────────────────────────┘
```

---

## Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation.
100 engines run to failure, 21 sensors, 20,631 records. 10 constant sensors removed (variance < 0.01); 9 sensors with |correlation with RUL| > 0.60 selected as base features.

Source: NASA Prognostics Center of Excellence Data Repository. Dataset used for academic purposes under its public-domain terms.

---

## Repository Contents

| File | Description |
|---|---|
| `app.py` | Streamlit web application (4 tabs: Upload CSV, Manual Input, Model Performance, Project Journey) |
| `requirements.txt` | Python dependencies (scikit-learn pinned to 1.8.0 to match the saved models) |
| `isolation_forest_v3.pkl` | Trained Isolation Forest anomaly detector |
| `rul_regressor.pkl` | Trained Gradient Boosting RUL regressor |
| `scaler_anomaly.pkl` | StandardScaler fitted for the anomaly model |
| `scaler_regression.pkl` | StandardScaler fitted for the regression model |
| `model_config.json` | Feature list, RUL cap, and validated metrics |
| `sample_engine_data.csv` | Sample 6-engine dataset for testing the upload feature |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/arafat077-dev/jet-engine-monitor.git
cd jet-engine-monitor

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens in your browser. Go to the **Upload CSV** tab and upload `sample_engine_data.csv` to see the full analysis.

### Expected CSV format

A CSV with these columns: `engine_id`, `cycle`, and the 9 sensors `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_11`, `sensor_12`, `sensor_17`, `sensor_20`, `sensor_21`.

---

## Tech Stack

Python · scikit-learn · pandas · NumPy · Plotly · Streamlit
Developed and tested in Google Colab; deployed on Streamlit Community Cloud.

---

## Team — Group 2

| Member | Role |
|---|---|
| Arafat Mohammed | Project Lead & Lead ML Engineer · web application developer |
| Latipov Javokhir Eldor Ugli | Lead ML Engineer |
| Bhuiyan Ahasanul Monir | Systems Architect |
| Esha Anika Tajnima | QA & Testing Engineer |
| Yusupjonov Otabek Odiljon Ugli | Decision & Risk Analyst |
| Rubayed | Data Visualisation Specialist |
| Arfin Ifthekhar | Technical Writer |

---

## Honest Limitations

This is an academic prototype. It is trained and validated only on the simulated C-MAPSS FD001 dataset; the RUL regressor is weaker for abruptly-degrading engines; and per-prediction explainability is not yet implemented. Real-world deployment would require validation on real engine data and additional operating conditions (FD002–FD004).

---

*Developed for the Basic Design course at Sejong University. The web application was designed and built by Arafat Mohammed.*
