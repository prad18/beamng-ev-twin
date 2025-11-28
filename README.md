# BeamNG EV Battery Digital Twin

> **A real-time battery degradation simulation system connecting BeamNG.tech driving physics with PyBaMM electrochemical modeling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![BeamNG.tech](https://img.shields.io/badge/BeamNG.tech-v0.36-orange.svg)](https://beamng.tech/)
[![PyBaMM](https://img.shields.io/badge/PyBaMM-electrochemistry-green.svg)](https://pybamm.org/)

---

## 🎯 Project Overview

This project creates a **Digital Twin** of an electric vehicle battery by combining:

1. **BeamNG.tech** — Realistic vehicle physics and driving simulation
2. **PyBaMM** — Physics-based electrochemical battery modeling (DFN model)
3. **Real-time Dashboard** — Live visualization of battery state

The system extracts motor torque, speed, and power data from BeamNG driving simulations, feeds it into a scientific battery model, and predicts realistic degradation over time.

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   BeamNG.tech   │ ───► │   PyBaMM API    │ ───► │    Dashboard    │
│  (Driving Sim)  │      │  (Battery Twin) │      │  (Streamlit)    │
│                 │      │                 │      │                 │
│ • Motor torque  │      │ • SOC/SOH calc  │      │ • Live gauges   │
│ • Vehicle speed │      │ • Degradation   │      │ • History plots │
│ • Regen braking │      │ • Temperature   │      │ • Reports       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 🔋 What This Project Does

### The Problem
Real EV batteries degrade over years of use. Testing this in the real world is:
- **Expensive** — Real batteries cost $10,000+
- **Slow** — Degradation takes months/years to observe
- **Limited** — Can't test extreme scenarios safely

### Our Solution
We simulate **realistic driving patterns** in BeamNG and use **PyBaMM's electrochemical equations** to predict battery degradation in **accelerated time** (10,000x speed).

**15 minutes of simulation = ~100 days of battery aging**

### Key Features
- ✅ Real-time telemetry extraction from BeamNG electric vehicles
- ✅ Physics-based battery modeling (not guesswork)
- ✅ Doyle-Fuller-Newman (DFN) electrochemical model
- ✅ SEI layer growth degradation modeling
- ✅ Temperature-dependent aging (Arrhenius equation)
- ✅ Live Streamlit dashboard with gauges and plots
- ✅ Automated stress testing with aggressive driving patterns
- ✅ Report generation for analysis

---

## 🏗️ Project Structure

```
beamng-ev-twin/
├── twin_service/           # Battery Digital Twin Backend
│   ├── api_pybamm.py       # FastAPI server (main API)
│   ├── pybamm_model.py     # PyBaMM electrochemical model
│   └── api.py              # Simple fallback model
│
├── main_files/             # Main Application Scripts
│   ├── demo_simulation.py  # Interactive BeamNG + PyBaMM demo
│   ├── stress_test_auto.py # Automated 15-min stress test
│   ├── streamlit_dashboard.py # Real-time monitoring UI
│   ├── config.yaml         # Configuration file
│   └── live_data.json      # Real-time data exchange
│
├── beamng_client/          # BeamNG Integration Scripts
│   ├── beamng_telemetry.py # Telemetry extraction
│   ├── test_battery_sensors.py # Sensor testing
│   └── config.yaml         # BeamNG paths
│
├── battery_sensor_data/    # Collected Sensor Data
│   └── *.json              # Raw telemetry exports
│
├── reports/                # Generated Test Reports
│   └── stress_test_*.txt   # Human-readable reports
│
├── datasets/               # External Dataset Tools
│   ├── download_nasa.py    # NASA battery dataset
│   └── download_stanford.py # Stanford dataset
│
└── ml_models/              # Machine Learning (WIP)
    └── train_degradation_model.py
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- BeamNG.tech v0.36+ (with valid license)
- 8GB+ RAM recommended

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure BeamNG Path
Edit `main_files/config.yaml`:
```yaml
beamng_home: "D:/BeamNG.tech.v0.36.4.0"
```

### 3. Start the PyBaMM API Server
```bash
cd twin_service
python api_pybamm.py
```
Server runs at `http://127.0.0.1:8008`

### 4. Run the Demo Simulation
```bash
cd main_files
python demo_simulation.py
```

### 5. (Optional) Open the Dashboard
```bash
cd main_files
streamlit run streamlit_dashboard.py
```

---

## 📊 Running a Stress Test

The stress test runs aggressive driving patterns and generates a detailed report:

```bash
cd main_files
python stress_test_auto.py --duration 15
```

**Driving Patterns Tested:**
- Full throttle acceleration
- Hard braking
- Throttle spamming (worst case)
- Regenerative coasting
- Repeated acceleration cycles

**Output:** Reports saved to `reports/stress_test_YYYYMMDD_HHMMSS.txt`

---

## 🔬 Technical Details

### Battery Specifications (Kia EV3)
| Parameter | Value |
|-----------|-------|
| Chemistry | LiFePO₄ (LFP) |
| Capacity | 81.4 kWh / 220 Ah |
| Nominal Voltage | 370V |
| Max Discharge | 185 kW |
| Max Regen | 92.5 kW |

### PyBaMM Model
We use the **Doyle-Fuller-Newman (DFN)** model, which solves:
- Lithium diffusion in electrode particles
- Ion transport in electrolyte
- Butler-Volmer reaction kinetics
- Heat generation and thermal dynamics

### Degradation Modeling
Battery aging is calculated using:
- **SEI layer growth** — Main capacity fade mechanism
- **Arrhenius temperature dependence** — Heat accelerates aging
- **C-rate stress** — High currents increase degradation
- **SOC stress factors** — Extreme SOC (< 20% or > 80%) causes extra wear

---

## 📈 Sample Results

From a 15-minute stress test (simulating ~100 days):

```
SIMULATION SUMMARY
------------------
  Real Duration:       15.0 minutes
  Simulated Time:      103.5 days
  Time Acceleration:   10,000x

BATTERY RESULTS
---------------
  Starting SOH:        100.0%
  Final SOH:           99.12%
  Degradation:         0.88%
  Equivalent Cycles:   621.4
  
STRESS METRICS
--------------
  Distance Driven:     45.2 km
  Energy Used:         18.7 kWh
  Peak Power:          142.3 kW
  Max Temperature:     38.4°C
```

---

## 🛠️ API Endpoints

The PyBaMM API server (`twin_service/api_pybamm.py`) provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check, returns model type |
| `/info` | GET | Battery specs and current state |
| `/step` | POST | Simulate one timestep |
| `/reset` | POST | Reset battery to fresh state |

### Example `/step` Request
```json
POST /step
{
  "current_A": 150.0,
  "soc": 0.80,
  "temperature_C": 35.0,
  "dt_s": 1.0,
  "accelerated_dt_s": 10000.0
}
```

---

## 📚 Documentation

- [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) — Full technical documentation
- [`PYBAMM_INTEGRATION.md`](PYBAMM_INTEGRATION.md) — PyBaMM integration guide
- [`battery_sensor_data/SENSOR_DATA_DOCUMENTATION.md`](battery_sensor_data/SENSOR_DATA_DOCUMENTATION.md) — Sensor data format
- [`EV_Setup_Guide.md`](EV_Setup_Guide.md) — BeamNG EV setup instructions

---

## 🙏 Acknowledgments

- **[BeamNG.tech](https://beamng.tech/)** — Vehicle simulation platform (Academic License)
- **[PyBaMM](https://pybamm.org/)** — Battery modeling framework
- **[beamngpy](https://github.com/BeamNG/BeamNGpy)** — Python API for BeamNG

---

## 📄 License

This project is for academic/research purposes. BeamNG.tech requires a valid license.

---

 
