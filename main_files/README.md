# 🔋 EV Battery Digital Twin - Main Implementation

This folder contains the core implementation of the BeamNG EV Battery Digital Twin system. This documentation covers all components, their interactions, and how to use them.

---

## 📁 File Structure

```
main_files/
├── demo_simulation.py      # 🎮 Main BeamNG simulation with accelerated degradation
├── streamlit_dashboard.py  # 📊 Real-time Streamlit dashboard (recommended)
├── dashboard.py            # 🖥️ Terminal-based ASCII dashboard (fallback)
├── simulation.py           # 🔧 Alternative simulation with twin service
├── battery_sensors.py      # 🔍 Battery sensor analyzer/explorer tool
├── current_model.py        # ⚡ Pack current estimation utilities
├── config.yaml             # ⚙️ Configuration file
├── live_data.json          # 📄 Real-time data exchange (auto-generated)
└── README.md               # 📖 This documentation
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BeamNG.tech Simulator                              │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│   │   Kia EV3       │    │   Tesla Model 3 │    │   Other EVs     │     │
│   │   (sv1ev3)      │    │   (Model3_2024) │    │   (custom)      │     │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│            │                      │                      │              │
│            └──────────────────────┼──────────────────────┘              │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                      │
│                    │ Electrics + PowertrainSensor│                      │
│                    │  (motor torque, speed, etc.)│                      │
│                    └──────────────┬───────────────┘                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │   demo_simulation.py      │
                      │   ────────────────────    │
                      │   • BatteryState class    │
                      │   • LiveDemoSimulation    │
                      │   • Time acceleration     │
                      │   • Degradation physics   │
                      └─────────────┬─────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │   live_data.json          │
                      │   (real-time exchange)    │
                      └─────────────┬─────────────┘
                                    │
          ┌────────────────────────┬┴─────────────────────────┐
          │                        │                          │
┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌────────────▼───────────┐
│ streamlit_dashboard│   │   dashboard.py    │   │  twin_service/api.py   │
│    (Recommended)   │   │   (Terminal)      │   │  (FastAPI server)      │
│                    │   │                   │   │                        │
│ • Plotly gauges    │   │ • ASCII art       │   │ • LiFePO4 model        │
│ • Real-time charts │   │ • Terminal UI     │   │ • Thermal model        │
│ • Browser-based    │   │ • No dependencies │   │ • REST API             │
└────────────────────┘   └───────────────────┘   └────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install beamngpy streamlit plotly pandas pyyaml requests
```

### Running the Demo

**Terminal 1 - Start BeamNG Simulation:**
```bash
cd main_files
python demo_simulation.py
```

**Terminal 2 - Start Streamlit Dashboard:**
```bash
cd main_files
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

---

## 📚 Component Details

### 1. `demo_simulation.py` - Main Simulation Engine

The core simulation script that connects to BeamNG and runs battery degradation modeling.

#### Key Classes

##### `BatteryState`
Models battery physics with accelerated degradation for demos.

```python
class BatteryState:
    # Battery Specifications
    capacity_ah = 220      # Amp-hours (LiFePO4)
    nominal_voltage = 370  # Volts
    energy_kwh = 81.4      # Total energy capacity

    # Real-time State
    soc: float         # State of Charge (0-1)
    soh: float         # State of Health (0-1)
    temperature: float # Cell temperature (°C)
    voltage: float     # Pack voltage
    current: float     # Pack current (A)
    power_kw: float    # Power (kW)
```

**Degradation Model:**
| Factor | Formula | Description |
|--------|---------|-------------|
| Cycle Aging | `2e-8 × C_rate^1.5 × dt` | Wear from charge/discharge cycles |
| Thermal Aging | `1e-9 × exp((T-25)/15) × dt` | Arrhenius-based temperature degradation |
| Calendar Aging | `5e-10 × SOC_stress × dt` | Time-based aging (even when idle) |

**Time Acceleration:**
- Demo Mode: 10,000x multiplier (15 min real = ~100 days simulated)
- Realistic Mode: 1x multiplier (real-time)

##### `LiveDemoSimulation`
Manages BeamNG connection and sensor data extraction.

```python
class LiveDemoSimulation:
    def connect()          # Connect to BeamNG, load vehicle
    def poll_sensors()     # Get motor torque, speed from PowertrainSensor
    def update_battery()   # Apply physics and degradation
    def save_state()       # Write to live_data.json
```

**Sensor Data Extracted:**
| Sensor | Field | Description |
|--------|-------|-------------|
| PowertrainSensor | `outputTorque1` | Motor output torque (Nm) |
| PowertrainSensor | `outputAV1` | Motor angular velocity (rad/s) |
| Electrics | `wheelspeed` | Vehicle wheel speed (m/s) |
| Electrics | `throttle` | Throttle position (0-1) |

---

### 2. `streamlit_dashboard.py` - Real-time Dashboard

Beautiful browser-based dashboard with Plotly visualizations.

#### Features
- **4 Gauge Charts**: SOC, SOH, Power, Temperature
- **Real-time History**: SOH/SOC line chart, Power area chart
- **Statistics Panel**: Degradation rate, 2-year prediction, years to 80% SOH
- **Status Indicators**: Running state, vehicle model, demo mode
- **Auto-refresh**: Configurable 0.5-5 second refresh rate

#### Usage
```bash
streamlit run streamlit_dashboard.py
```

#### Key Functions
```python
create_gauge(value, title, ...)     # Create Plotly gauge chart
create_power_gauge(power)           # Power gauge with regen support
create_history_chart(time, soh, soc) # SOH/SOC history line chart
create_power_history(time, power)   # Power flow area chart
```

---

### 3. `dashboard.py` - Terminal Dashboard

ASCII-based terminal dashboard for environments without browser access.

#### Features
- Progress bar visualization
- Color-coded status (requires colorama)
- 4 updates per second
- No external dependencies beyond colorama

---

### 4. `simulation.py` - Twin Service Integration

Alternative simulation that integrates with the FastAPI twin service.

#### Architecture
```
BeamNG → simulation.py → twin_service/api.py → LiFePO4 Model
                                            → Thermal Model
```

#### Usage
```bash
# Terminal 1: Start twin service
cd twin_service
uvicorn api:app --port 8008

# Terminal 2: Run simulation
cd main_files
python simulation.py
```

---

### 5. `battery_sensors.py` - Sensor Explorer

Interactive tool for discovering available battery data from any EV vehicle.

#### Features
- User input for vehicle selection
- Automatic sensor attachment
- Lua query integration
- Organized data storage in `battery_sensor_data/` folder

#### Usage
```bash
python battery_sensors.py
# Enter vehicle model: sv1ev3
# Results saved to: battery_sensor_data/sv1ev3_*.json
```

---

### 6. `current_model.py` - Current Estimation

Utility module for estimating pack current from electrical power.

```python
def est_pack_current(power_kw, voltage_nom=360.0, eff=0.94):
    """
    Estimate pack current from electrical power.
    
    Args:
        power_kw: Electrical power (kW), positive = discharge
        voltage_nom: Nominal pack voltage
        eff: Drivetrain efficiency (94% default)
    
    Returns:
        Pack current in Amps
    """
```

---

### 7. `config.yaml` - Configuration

```yaml
beamng_home: "D:/BeamNG.tech.v0.36.4.0"  # BeamNG installation path
twin_url: "http://127.0.0.1:8008/step"    # Twin service endpoint
```

---

### 8. `live_data.json` - Data Exchange

Auto-generated file for real-time communication between simulation and dashboard.

#### Schema
```json
{
    "timestamp": "2025-11-26T12:00:00",
    "elapsed_seconds": 120.5,
    "simulated_days": 14.2,
    "simulated_years": 0.039,
    
    "soc": 78.5,
    "soh": 99.7,
    "temperature": 28.3,
    "voltage": 368.2,
    "current": 45.2,
    "power_kw": 16.6,
    
    "total_distance_km": 12.5,
    "cycle_count": 0.15,
    "energy_throughput_kwh": 12.3,
    
    "is_charging": false,
    "is_regen": false,
    "demo_mode": true,
    "time_multiplier": 10000,
    
    "vehicle_model": "sv1ev3",
    "simulation_running": true
}
```

---

## 🎮 Supported Vehicles

| Vehicle | Model ID | Motor Config | Notes |
|---------|----------|--------------|-------|
| Kia EV3 | `sv1ev3` | Rear motor | Primary test vehicle |
| Tesla Model 3 | `Model3_2024` | Rear motor | `rearMotor.outputTorque1` |
| Other EVs | varies | varies | Use `battery_sensors.py` to explore |

---

## 🔬 Physics Models

### Battery Degradation (Simplified)

The degradation model combines three aging mechanisms:

1. **Cycle Aging** (stress from use)
   ```
   Δ_cycle = k₁ × C_rate^1.5 × Δt
   ```
   - Higher C-rate (fast charge/discharge) = more degradation
   - Exponent 1.5 reflects accelerated wear at high rates

2. **Thermal Aging** (Arrhenius)
   ```
   Δ_thermal = k₂ × exp((T - 25) / 15) × Δt
   ```
   - Exponential increase with temperature
   - Reference temperature: 25°C

3. **Calendar Aging** (time-based)
   ```
   Δ_calendar = k₃ × SOC_factor × Δt
   ```
   - SOC factor: 1.5 if SOC > 80%, 1.3 if SOC < 20%, else 1.0
   - High/low SOC accelerates aging

### Thermal Model (Lumped)

```
dT/dt = (I² × R_int - h × (T - T_amb)) / (m × Cp)
```
- Heat generation: I²R losses
- Cooling: Natural convection to ambient
- Thermal mass: Cell mass × specific heat

---

## 📊 Demo Mode Timeline

With 10,000x acceleration:

| Real Time | Simulated | Expected SOH |
|-----------|-----------|--------------|
| 2 min | ~14 days | 99.8% |
| 5 min | ~35 days | 99.5% |
| 10 min | ~70 days | 99.0% |
| 15 min | ~105 days | 98.5% |
| 30 min | ~210 days | 97.0% |
| 1 hour | ~420 days | 95.0% |

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `beamngpy not found` | `pip install beamngpy` |
| Connection refused | Ensure BeamNG.tech is running |
| Vehicle not loading | Check vehicle model name in BeamNG |
| No sensor data | Use `battery_sensors.py` to debug |
| Dashboard not updating | Check `live_data.json` is being written |

### Debug Commands

```bash
# Test BeamNG connection
python -c "from beamngpy import BeamNGpy; b = BeamNGpy('localhost', 64256); b.open(); print('OK')"

# Check live data
cat live_data.json | python -m json.tool

# Monitor live data updates
watch -n 1 cat live_data.json
```

---

## 🔮 Future Enhancements

- [ ] ML-based degradation prediction (LSTM/Transformer)
- [ ] AMESIM/Simulink co-simulation integration
- [ ] Multi-vehicle fleet simulation
- [ ] Cloud-based twin service deployment
- [ ] Historical data analytics dashboard
- [ ] OTA update impact modeling

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2025 | Initial implementation with BeamNG integration |
| 1.1 | Nov 2025 | Added Streamlit dashboard |
| 1.2 | Nov 2025 | Time acceleration for demos |

---

## 👥 Contributors

Built for the EV Battery Digital Twin project. See main project `PROJECT_DOCUMENTATION.md` for team details.

---

*Last updated: November 2025*
