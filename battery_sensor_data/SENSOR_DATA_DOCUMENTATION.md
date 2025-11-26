# 🔋 Battery Sensor Data Documentation

This document describes the data output from `battery_sensors.py` for BeamNG EV vehicles. The tool collects data from three sources and saves them in organized JSON files.

---

## 📁 Output Files

For each vehicle tested, three JSON files are generated:

| File Pattern | Description |
|--------------|-------------|
| `{vehicle}_electrics_{timestamp}.json` | Electrics sensor data (vehicle systems) |
| `{vehicle}_powertrain_{timestamp}.json` | Powertrain sensor data (motors, drivetrain) |
| `{vehicle}_battery_analysis_{timestamp}.json` | Combined analysis with metadata |

**Location:** `battery_sensor_data/` folder

---

## 🚗 Tested Vehicles

| Vehicle Model | Model ID | Motor Config | Files Available |
|---------------|----------|--------------|-----------------|
| Kia EV3 | `sv1ev3` | Front motor (FWD) | ✅ Full data |
| Porsche Macan EV | `newmacanthenotorious` | Dual motor (AWD) | ✅ Full data |
| Tesla Model 3 | `Model3_2024` | Rear motor (RWD) | ✅ Full data |

---

## ⚡ Electrics Sensor Data

The Electrics sensor provides real-time vehicle electrical system data.

### Battery & Energy Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `fuel` | float | 0.0 - 1.0 | Battery charge level (SOC equivalent) |
| `fuel_volume` | float | kWh | Current energy in battery |
| `fuel_capacity` | float | kWh | Total battery capacity |
| `lowfuel` | bool | true/false | Low battery warning |

**Example (sv1ev3):**
```json
{
  "fuel": 1.0,
  "fuel_volume": 81.4,
  "fuel_capacity": 81.4,
  "lowfuel": false
}
```

### Regenerative Braking Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `regenThrottle` | float | 0.0 - 1.0 | Regen from throttle lift-off |
| `regenFromOnePedal` | float | 0.0 - 1.0 | Regen from one-pedal driving mode |
| `regenFromBrake` | float | 0.0 - 1.0 | Regen from brake pedal |
| `regenStrength` | float | 0 - 3 | Current regen strength level |
| `maxRegenStrength` | float | 3.0 | Maximum regen level available |

**Example:**
```json
{
  "regenThrottle": 0.0,
  "regenFromOnePedal": 0.0,
  "regenFromBrake": 0.0,
  "regenStrength": 3.0,
  "maxRegenStrength": 3.0
}
```

### Motor & Drivetrain Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `rpm` | float | 0 - 13000 | Motor RPM |
| `maxrpm` | float | varies | Maximum motor RPM |
| `idlerpm` | float | 0.0 | Idle RPM (0 for EVs) |
| `engineRunning` | float | 0/1 | Motor active status |
| `gear` | string | "N", "D", "R" | Current gear |
| `gearModeIndex` | float | 0-3 | Gear mode selector position |
| `gearboxMode` | string | "arcade" | Transmission mode |

**Example:**
```json
{
  "rpm": 0.302,
  "maxrpm": 13000.0,
  "idlerpm": 0.0,
  "engineRunning": 1.0,
  "gear": "N",
  "gearModeIndex": 3.0,
  "gearboxMode": "arcade"
}
```

### Vehicle Speed & Motion

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `wheelspeed` | float | m/s | Average wheel speed |
| `airspeed` | float | m/s | Vehicle airspeed |
| `airflowspeed` | float | m/s | Air flow over vehicle |
| `virtualAirspeed` | float | m/s | Calculated airspeed |
| `altitude` | float | m | Vehicle altitude |
| `odometer` | float | km | Total distance traveled |
| `trip` | float | km | Trip distance |

**Example:**
```json
{
  "wheelspeed": 0.00116,
  "airspeed": 0.00395,
  "odometer": 0.0352,
  "trip": 0.0352,
  "altitude": 0.2017
}
```

### Driver Inputs

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `throttle` | float | 0.0 - 1.0 | Current throttle position |
| `throttle_input` | float | 0.0 - 1.0 | Raw throttle input |
| `brake` | float | 0.0 - 1.0 | Current brake position |
| `brake_input` | float | 0.0 - 1.0 | Raw brake input |
| `steering` | float | -1.0 - 1.0 | Steering angle |
| `steering_input` | float | -1.0 - 1.0 | Raw steering input |
| `parkingbrake` | float | 0/1 | Parking brake state |
| `parkingbrake_input` | float | 0/1 | Parking brake input |

**Example:**
```json
{
  "throttle": 0.0,
  "throttle_input": 0.0,
  "brake": 0.3,
  "brake_input": 0.0,
  "steering": 3.83e-10,
  "parkingbrake": 1.0
}
```

### Stability Systems

| Field | Type | Description |
|-------|------|-------------|
| `hasABS` | float | ABS equipped (1=yes) |
| `abs` | float | ABS active level |
| `abs_active` | float | ABS currently active |
| `isABSBrakeActive` | float | ABS brake intervention |
| `hasESC` | float | ESC equipped (1=yes) |
| `esc` | float | ESC active level |
| `esc_active` | bool | ESC currently active |
| `hasTCS` | float | TCS equipped (1=yes) |
| `tcs` | float | TCS active level |
| `tcs_active` | bool | TCS currently active |

### Acceleration (Smoothed)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `accXSmooth` | float | m/s² | Longitudinal acceleration |
| `accYSmooth` | float | m/s² | Lateral acceleration |
| `accZSmooth` | float | m/s² | Vertical acceleration (gravity ~-9.8) |

### Wheel Thermal Data

Brake temperature data for each wheel:

```json
{
  "wheelThermals": {
    "FL": {
      "brakeSurfaceTemperature": 15.0,
      "brakeCoreTemperature": 15.0,
      "brakeThermalEfficiency": 0.959
    },
    "FR": { ... },
    "RL": { ... },
    "RR": { ... }
  }
}
```

| Field | Unit | Description |
|-------|------|-------------|
| `brakeSurfaceTemperature` | °C | Brake pad surface temp |
| `brakeCoreTemperature` | °C | Brake rotor core temp |
| `brakeThermalEfficiency` | 0-1 | Brake efficiency (heat fade) |

---

## ⚙️ Powertrain Sensor Data

The PowertrainSensor provides detailed drivetrain component data with time-series measurements.

### Data Structure

```json
{
  "0.0": { /* timestamp 0.0 seconds */ },
  "1.0": { /* timestamp 1.0 seconds */ },
  "2.0": { /* timestamp 2.0 seconds */ },
  ...
}
```

Each timestamp contains data for all powertrain components.

### Motor Data (Primary for Battery Simulation)

#### Front Motor (FWD vehicles like sv1ev3)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `frontMotor.outputTorque1` | float | Nm | Motor output torque |
| `frontMotor.outputAV1` | float | rad/s | Motor angular velocity |
| `frontMotor.inputAV` | float | rad/s | Input shaft angular velocity |
| `frontMotor.gearRatio` | float | - | Fixed gear ratio |
| `frontMotor.isBroken` | bool | - | Motor failure state |

**Example (sv1ev3 front motor):**
```json
{
  "frontMotor": {
    "isBroken": false,
    "outputAV1": 0.0258,
    "parentOutputIndex": 0.0,
    "gearRatio": 1.0,
    "outputTorque1": -2.0,
    "inputAV": 0.0
  }
}
```

#### Rear Motor (RWD vehicles like Model3_2024)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `rearMotor.outputTorque1` | float | Nm | Motor output torque |
| `rearMotor.outputAV1` | float | rad/s | Motor angular velocity |

### Power Calculation

To calculate battery power from motor data:

```python
# Motor power (mechanical)
power_kw = (outputTorque1 * outputAV1) / 1000

# Positive torque + positive AV = discharge (driving)
# Negative torque + positive AV = regen (braking)
```

### Differential Data

| Field | Description |
|-------|-------------|
| `differential_F.outputAV1` | Left output angular velocity |
| `differential_F.outputAV2` | Right output angular velocity |
| `differential_F.outputTorque1` | Left output torque |
| `differential_F.outputTorque2` | Right output torque |
| `differential_F.mode` | Differential mode (open, locked, etc.) |

**Example:**
```json
{
  "differential_F": {
    "outputAV1": 0.00312,
    "outputAV2": 0.00338,
    "gearRatio": 1.0,
    "mode": "open",
    "parentName": "torsionReactorF",
    "outputTorque1": -7.82,
    "outputTorque2": -7.82
  }
}
```

### Torsion Reactor (Gear Reduction)

| Field | Description |
|-------|-------------|
| `torsionReactorF.gearRatio` | Final drive ratio (e.g., 7.94:1 for sv1ev3) |
| `torsionReactorF.outputTorque1` | Output torque after gear reduction |

**Example:**
```json
{
  "torsionReactorF": {
    "gearRatio": 7.94,
    "outputTorque1": -15.88,
    "parentName": "frontMotor"
  }
}
```

### Wheel Axle Data

| Field | Description |
|-------|-------------|
| `wheelaxleFL.outputAV1` | Front-left wheel angular velocity |
| `wheelaxleFL.outputTorque1` | Front-left wheel torque |
| `wheelaxleFR.outputAV1` | Front-right wheel angular velocity |
| `spindleFL`, `spindleFR`, etc. | Wheel spindle data |

### Disconnected Components (RWD/FWD)

For single-motor vehicles, non-driven wheels show:
```json
{
  "spindleRL": {
    "mode": "disconnected",
    "inputAV": 0.0,
    "outputTorque1": 0.0
  }
}
```

---

## 📊 Battery Analysis File

The `_battery_analysis_*.json` file combines all data with metadata:

```json
{
  "timestamp": "2025-09-15T11:03:11.832516",
  "vehicle_model": "sv1ev3",
  "vehicle_name": "sv1ev3",
  "test_description": "Battery sensor data analysis for degradation testing",
  
  "electrics_data": { /* full electrics data */ },
  "electrics_battery_fields": { /* filtered battery-related fields */ },
  
  "powertrain_data": { /* time-series powertrain data */ },
  
  "lua_battery_data": { /* Lua query results */ },
  
  "battery_analysis": {
    "motor_config": "front",
    "regen_capable": true,
    "battery_capacity_kwh": 81.4
  }
}
```

---

## 🔍 Key Fields for Battery Simulation

### Essential Fields for Degradation Modeling

| Source | Field | Use |
|--------|-------|-----|
| Electrics | `fuel` | SOC tracking |
| Electrics | `fuel_volume` / `fuel_capacity` | Energy calculations |
| Electrics | `regenThrottle`, `regenFromBrake` | Regen event detection |
| Electrics | `wheelspeed` | Distance tracking |
| Powertrain | `frontMotor.outputTorque1` | Power calculation |
| Powertrain | `frontMotor.outputAV1` | Power calculation |

### Power Flow Calculation

```python
def calculate_power(torque_nm, angular_velocity_rad_s):
    """
    Calculate electrical power from motor data.
    
    Returns:
        Power in kW (positive = discharge, negative = regen)
    """
    return (torque_nm * angular_velocity_rad_s) / 1000.0
```

### C-Rate Estimation

```python
def estimate_c_rate(power_kw, battery_capacity_kwh=81.4, voltage=370):
    """
    Estimate C-rate from power.
    
    C-rate of 1.0 = full discharge in 1 hour
    """
    current_a = (power_kw * 1000) / voltage
    capacity_ah = (battery_capacity_kwh * 1000) / voltage
    return abs(current_a) / capacity_ah
```

---

## 🚗 Vehicle-Specific Notes

### Kia EV3 (sv1ev3)

- **Motor:** Front motor (FWD)
- **Battery:** 81.4 kWh
- **Gear Ratio:** 7.94:1
- **Max RPM:** 13,000
- **Regen Levels:** 0-3
- **Key Fields:** `frontMotor.outputTorque1`, `frontMotor.outputAV1`

### Porsche Macan EV (newmacanthenotorious)

- **Motor:** Dual motor (AWD)
- **Battery:** ~100 kWh
- **Max RPM:** 16,000
- **Special Fields:** Extensive lighting system data, radio controls
- **Key Fields:** `frontMotor.*`, `rearMotor.*`

### Tesla Model 3 (Model3_2024)

- **Motor:** Rear motor (RWD)
- **Battery:** ~60-75 kWh (depending on variant)
- **Key Fields:** `rearMotor.outputTorque1`, `rearMotor.outputAV1`

---

## 📈 Sample Data Analysis

### Idle State (sv1ev3)

```
Motor Torque: -2.0 Nm (drag)
Motor Speed: 0.026 rad/s (~0.25 RPM)
Wheel Speed: 0.001 m/s
Power: -0.05 kW (minimal drain)
SOC: 100%
Temperature: 15°C (cold start)
```

### Driving State (typical)

```
Motor Torque: 150 Nm
Motor Speed: 500 rad/s (~4775 RPM)
Wheel Speed: 20 m/s (72 km/h)
Power: 75 kW
C-Rate: 0.92
SOC: decreasing
Temperature: 35°C
```

### Regenerative Braking

```
Motor Torque: -80 Nm (negative = regen)
Motor Speed: 300 rad/s
Power: -24 kW (negative = charging)
regenFromBrake: 0.5
regenStrength: 3
SOC: slowly increasing
```

---

## 🔧 Using the Data

### Load Data in Python

```python
import json

# Load electrics data
with open('battery_sensor_data/sv1ev3_electrics_20250915_110335.json') as f:
    electrics = json.load(f)

# Get SOC
soc = electrics['fuel']  # 0.0 - 1.0

# Get battery capacity
capacity_kwh = electrics['fuel_capacity']

# Load powertrain time-series
with open('battery_sensor_data/sv1ev3_powertrain_20250915_110335.json') as f:
    powertrain = json.load(f)

# Iterate through timestamps
for timestamp, data in powertrain.items():
    if 'frontMotor' in data:
        torque = data['frontMotor']['outputTorque1']
        speed = data['frontMotor']['outputAV1']
        power_kw = (torque * speed) / 1000
        print(f"t={timestamp}s: {power_kw:.2f} kW")
```

### Extract Battery Events

```python
def detect_regen_events(electrics_data):
    """Detect regenerative braking events."""
    events = []
    if electrics_data.get('regenFromBrake', 0) > 0.1:
        events.append('brake_regen')
    if electrics_data.get('regenFromOnePedal', 0) > 0.1:
        events.append('one_pedal_regen')
    if electrics_data.get('regenThrottle', 0) > 0.1:
        events.append('coast_regen')
    return events
```

---

## 🤖 ML Model Integration

This section covers how to use the BeamNG sensor data with machine learning models for battery degradation prediction.

### ML Feasibility Assessment

#### ✅ Available Public Datasets

| Dataset | Size | Source | Best For |
|---------|------|--------|----------|
| **NASA Battery Dataset** | 34 cells | NASA PCoE | Capacity fade, impedance growth |
| **CALCE Battery Data** | 100+ cells | U of Maryland | LiFePO4, NMC cycling profiles |
| **Oxford Battery Degradation** | 8 cells | Oxford University | Long-term aging with temperature |
| **Stanford/MIT Fast Charging** | 124 cells | Nature Energy | Fast charging degradation |

**Download Links:**
- NASA: `https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/`
- CALCE: `https://calce.umd.edu/battery-data`

#### BeamNG Data → ML Feature Mapping

| BeamNG Sensor Field | ML Feature | How to Calculate |
|---------------------|------------|------------------|
| `frontMotor.outputTorque1` × `outputAV1` | **Power (kW)** | `torque × speed / 1000` |
| Power / (Voltage × Capacity) | **C-rate** | See code below |
| `fuel` (start - end of cycle) | **Depth of Discharge** | `SOC_start - SOC_end` |
| `wheelThermals.*.brakeCoreTemperature` | **Temperature proxy** | Ambient + load factor |
| `regenFromBrake`, `regenThrottle` | **Regen cycles** | Count events > 0.1 |
| `odometer` / cycle | **Distance per cycle** | Cumulative tracking |

### Feature Engineering from Sensor Data

```python
import json
import numpy as np

def extract_ml_features(electrics_file, powertrain_file):
    """
    Extract ML-ready features from BeamNG sensor data.
    
    Returns features compatible with NASA/CALCE dataset training.
    """
    # Load data
    with open(electrics_file) as f:
        electrics = json.load(f)
    with open(powertrain_file) as f:
        powertrain = json.load(f)
    
    # Battery specs (Kia EV3)
    CAPACITY_KWH = electrics.get('fuel_capacity', 81.4)
    VOLTAGE_NOM = 370  # Nominal pack voltage
    CAPACITY_AH = (CAPACITY_KWH * 1000) / VOLTAGE_NOM
    
    features = {
        'soc': electrics.get('fuel', 1.0),
        'battery_capacity_kwh': CAPACITY_KWH,
        'regen_strength': electrics.get('regenStrength', 0),
        'max_regen_strength': electrics.get('maxRegenStrength', 3),
    }
    
    # Extract power profile from powertrain time-series
    powers = []
    c_rates = []
    
    for timestamp, data in powertrain.items():
        if timestamp == 'time':
            continue
            
        # Find motor data (front or rear)
        motor_data = data.get('frontMotor') or data.get('rearMotor')
        if motor_data:
            torque = motor_data.get('outputTorque1', 0)
            speed = motor_data.get('outputAV1', 0)
            power_kw = (torque * speed) / 1000
            powers.append(power_kw)
            
            # C-rate calculation
            current = (power_kw * 1000) / VOLTAGE_NOM
            c_rate = abs(current) / CAPACITY_AH
            c_rates.append(c_rate)
    
    if powers:
        features['avg_power_kw'] = np.mean(powers)
        features['max_power_kw'] = np.max(powers)
        features['min_power_kw'] = np.min(powers)  # Negative = regen
        features['avg_c_rate'] = np.mean(c_rates)
        features['max_c_rate'] = np.max(c_rates)
        features['regen_ratio'] = sum(1 for p in powers if p < 0) / len(powers)
    
    return features


def prepare_training_data(sensor_files, labels):
    """
    Prepare training dataset from multiple sensor recordings.
    
    Args:
        sensor_files: List of (electrics_file, powertrain_file) tuples
        labels: SOH values for each recording (from real dataset or simulation)
    
    Returns:
        X: Feature matrix
        y: SOH labels
    """
    X = []
    for electrics_file, powertrain_file in sensor_files:
        features = extract_ml_features(electrics_file, powertrain_file)
        X.append(list(features.values()))
    
    return np.array(X), np.array(labels)
```

### Training Pipeline with NASA Dataset

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load NASA dataset (example structure)
def load_nasa_battery_data(csv_path):
    """
    Load and preprocess NASA battery dataset.
    Expected columns: cycle, capacity, voltage, current, temperature, time
    """
    df = pd.read_csv(csv_path)
    
    # Calculate derived features
    df['soh'] = df['capacity'] / df['capacity'].iloc[0]  # Normalized SOH
    df['cumulative_ah'] = df['current'].abs().cumsum() * (df['time'].diff().fillna(0) / 3600)
    df['c_rate'] = df['current'].abs() / df['capacity'].iloc[0]
    df['dod'] = df.groupby('cycle')['capacity'].transform(lambda x: x.max() - x.min())
    
    return df


# Step 2: Feature engineering for ML
def create_cycle_features(df):
    """Aggregate per-cycle features."""
    cycle_features = df.groupby('cycle').agg({
        'soh': 'last',
        'c_rate': ['mean', 'max'],
        'temperature': ['mean', 'max'],
        'cumulative_ah': 'max',
        'dod': 'mean'
    }).reset_index()
    
    cycle_features.columns = ['cycle', 'soh', 'avg_c_rate', 'max_c_rate', 
                               'avg_temp', 'max_temp', 'cumulative_ah', 'dod']
    return cycle_features


# Step 3: Train model
def train_soh_predictor(features_df):
    """Train SOH prediction model."""
    feature_cols = ['cycle', 'avg_c_rate', 'max_c_rate', 'avg_temp', 
                    'max_temp', 'cumulative_ah', 'dod']
    
    X = features_df[feature_cols]
    y = features_df['soh']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    return model


# Step 4: Apply to BeamNG data
def predict_soh_from_beamng(model, electrics_file, powertrain_file, cycle_num):
    """
    Predict SOH using BeamNG sensor data.
    
    Maps BeamNG features to NASA-trained model features.
    """
    features = extract_ml_features(electrics_file, powertrain_file)
    
    # Map to model input format
    model_input = pd.DataFrame([{
        'cycle': cycle_num,
        'avg_c_rate': features.get('avg_c_rate', 0.5),
        'max_c_rate': features.get('max_c_rate', 1.0),
        'avg_temp': 30,  # Estimated from simulation
        'max_temp': 40,
        'cumulative_ah': cycle_num * 50,  # Rough estimate
        'dod': 0.8  # Typical DOD
    }])
    
    predicted_soh = model.predict(model_input)[0]
    return predicted_soh
```

### LSTM for Time-Series Prediction

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(sequence_length, n_features):
    """
    Build LSTM model for SOH prediction from time-series data.
    
    Input: sequences of [SOC, C-rate, Temperature, Power]
    Output: SOH prediction
    """
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # SOH output (0-1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def prepare_sequences(powertrain_data, sequence_length=50):
    """
    Convert BeamNG powertrain time-series to LSTM sequences.
    """
    sequences = []
    
    # Sort by timestamp
    timestamps = sorted([float(t) for t in powertrain_data.keys() if t != 'time'])
    
    for i in range(len(timestamps) - sequence_length):
        sequence = []
        for j in range(sequence_length):
            t = str(timestamps[i + j])
            data = powertrain_data[t]
            
            motor = data.get('frontMotor') or data.get('rearMotor', {})
            torque = motor.get('outputTorque1', 0)
            speed = motor.get('outputAV1', 0)
            power = (torque * speed) / 1000
            
            # Feature vector: [power, c_rate, estimated_temp, soc_proxy]
            c_rate = abs(power) / 81.4  # Normalized by capacity
            sequence.append([power, c_rate, 30, 0.8])  # Simplified
        
        sequences.append(sequence)
    
    return np.array(sequences)
```

### PyBaMM Integration (Recommended)

```python
# pip install pybamm

import pybamm

def create_battery_model():
    """
    Create a physics-based battery model using PyBaMM.
    Can be used alongside or instead of pure ML.
    """
    # Use Single Particle Model with degradation
    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited"}
    )
    
    # LiFePO4 parameters (approximate)
    param = pybamm.ParameterValues("Chen2020")
    
    # Create simulation
    sim = pybamm.Simulation(model, parameter_values=param)
    
    return sim


def simulate_degradation(sim, c_rate=1.0, cycles=100):
    """
    Simulate battery degradation over multiple cycles.
    
    Returns capacity fade curve for ML training augmentation.
    """
    experiment = pybamm.Experiment([
        (f"Discharge at {c_rate}C until 2.5V",
         "Rest for 1 hour",
         f"Charge at {c_rate/2}C until 3.6V",
         "Hold at 3.6V until C/50",
         "Rest for 1 hour")
    ] * cycles)
    
    sim_result = sim.solve(experiment)
    
    # Extract capacity fade
    capacity = sim_result["Discharge capacity [A.h]"].entries
    soh = capacity / capacity[0]
    
    return soh
```

### Recommended ML Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ML Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. DATA COLLECTION                                                 │
│     ├── NASA/CALCE datasets (real degradation data)                │
│     ├── PyBaMM simulations (synthetic augmentation)                 │
│     └── BeamNG sensor data (driving profiles)                       │
│                                                                     │
│  2. FEATURE ENGINEERING                                             │
│     ├── C-rate statistics (mean, max, std)                         │
│     ├── Temperature profiles                                        │
│     ├── Depth of Discharge patterns                                 │
│     ├── Cumulative Ah throughput                                    │
│     └── Regen event frequency                                       │
│                                                                     │
│  3. MODEL TRAINING                                                  │
│     ├── Baseline: RandomForest / XGBoost (fast, interpretable)     │
│     ├── Advanced: LSTM (time-series patterns)                       │
│     └── Physics-Informed: PINN (Arrhenius constraints)             │
│                                                                     │
│  4. BEAMNG INTEGRATION                                              │
│     ├── Real-time feature extraction from sensors                   │
│     ├── Model inference every N seconds                             │
│     └── Dashboard visualization                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Tools Comparison

| Tool | Cost | Use Case | Recommendation |
|------|------|----------|----------------|
| **PyBaMM** | Free | Physics-based simulation | ✅ Start here |
| **scikit-learn** | Free | Basic ML models | ✅ Use for baseline |
| **TensorFlow/PyTorch** | Free | Deep learning (LSTM) | ✅ For advanced models |
| **AMESIM** | $$$ | High-fidelity OEM simulation | ⚠️ Overkill for MVP |
| **ANSYS** | $$$ | Thermal CFD | ❌ Not needed yet |
| **Simulink** | $$ | System modeling | 🤔 Maybe for integration |

### Quick Start: 10-Day ML Pipeline

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Download NASA/CALCE data | Raw CSV files |
| 3-4 | Feature engineering | Training dataset |
| 5-6 | Train RandomForest baseline | Working model |
| 7-8 | BeamNG feature mapping | Integration code |
| 9-10 | Dashboard integration | Live SOH prediction |

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Sep 2025 | Initial sensor data collection |
| 1.1 | Nov 2025 | Added Macan EV and Model 3 data |
| 1.2 | Nov 2025 | Documentation created |
| 1.3 | Nov 2025 | Added ML integration guide |

---

*Generated from `battery_sensors.py` analysis tool*
