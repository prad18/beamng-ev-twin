# BeamNG EV Battery Digital Twin Project

## 🚗⚡ Overview

This project creates a **real-time battery simulation system** that connects BeamNG.tech driving simulator with a FastAPI-based battery digital twin service. It simulates realistic electric vehicle operation using a Kia EV3 model and tracks battery performance including degradation, thermal behavior, and state of charge.

## 🏗️ System Architecture

```
BeamNG.tech Simulator ←→ Python Client ←→ FastAPI Battery Twin ←→ Battery Physics Model
      (Kia EV3)              (BeamNG API)         (REST API)           (SOH/SOC/Thermal)
```

## 📁 Project Structure

```
beamng-ev-twin/
├── beamng_client/                 # BeamNG simulation client
│   ├── config.yaml               # Configuration file
│   ├── current_model.py          # Battery current estimation
│   ├── kia_ev3_simulation.py     # Main Kia EV3 simulation
│   ├── kia_working_test.py       # Quick Kia EV3 test
│   └── stream.py                 # Original basic simulation
├── twin_service/                 # Battery digital twin API
│   ├── api.py                    # FastAPI battery service
│   ├── battery_model.py          # Battery physics simulation
│   └── requirements.txt          # API dependencies
├── env/                          # Python virtual environment
├── kia_ev3_results_*.json        # Simulation results
└── README.md                     # This file
```

---

## 📋 File-by-File Code Explanation

### 🔧 Configuration Files

#### `beamng_client/config.yaml`
```yaml
# BeamNG.tech installation configuration
beamng_home: "C:/BeamNG.tech"    # Path to BeamNG installation
twin_url: "http://127.0.0.1:8008/step"  # Battery twin API endpoint
```
**Purpose**: Stores system configuration including BeamNG installation path and battery twin service URL.

---

### ⚡ Battery Current Estimation

#### `beamng_client/current_model.py`
```python
def est_pack_current(power_kw, pack_voltage=370.0):
    """
    Estimates battery pack current from electrical power
    
    Args:
        power_kw: Electrical power in kilowatts
        pack_voltage: Battery pack voltage (default: 370V for Kia EV3)
    
    Returns:
        current_a: Battery current in amperes
    """
    if power_kw == 0:
        return 0.0
    
    # P = V * I, therefore I = P / V
    current_a = (power_kw * 1000.0) / pack_voltage
    return current_a
```

**What it does**:
- Converts electrical power (kW) to battery current (A)
- Uses Ohm's law: Current = Power / Voltage
- Essential for battery twin communication

---

### 🚗 Main Kia EV3 Simulation

#### `beamng_client/kia_ev3_simulation.py`

**Key Components**:

##### 1. Vehicle Configuration
```python
def load_kia_ev3_config():
    """Loads Kia EV3 specifications"""
    return {
        'model_name': 'sv1ev3',              # BeamNG model name
        'battery_capacity_kwh': 81.4,        # Real Kia EV3 battery
        'battery_capacity_ah': 220.0,        # At 370V nominal
        'max_power_kw': 150.0,               # Motor power rating
        'efficiency': 0.85,                  # Motor efficiency
        'wheel_radius_m': 0.35,              # For speed calculations
        'mass_kg': 1850                      # Vehicle mass
    }
```

##### 2. Driving Patterns
```python
def get_driving_pattern(pattern_name, step):
    """Generates realistic power demands"""
    if pattern_name == 'aggressive':
        if step % 100 < 30:          # Acceleration phase
            return 150.0             # Max power
        elif step % 100 < 50:        # Cruise
            return 50.0              # Moderate power
        elif step % 100 < 60:        # Regen braking
            return -150.0            # Max regeneration
        else:                        # Coast
            return 15.0              # Low power
```

##### 3. Real-Time Simulation Loop
```python
def run_simulation_step(vehicle, step, pattern, soc, ambient_temp):
    """Single simulation step"""
    
    # 1. Get power demand from driving pattern
    target_power_kw = get_driving_pattern(pattern, step)
    
    # 2. Calculate battery current
    current_a = est_pack_current(target_power_kw)
    
    # 3. Update state of charge
    dt = 0.05  # 50ms timestep
    coulombs = current_a * dt
    ah_used = coulombs / 3600.0
    new_soc = soc - (ah_used / 220.0)  # 220Ah capacity
    
    # 4. Call battery twin service
    twin_response = requests.post('http://127.0.0.1:8008/step', json={
        "pack_current_A": float(current_a),
        "soc": float(new_soc),
        "amb_temp_C": float(ambient_temp),
        "dt_s": float(dt)
    })
    
    # 5. Get updated battery state
    battery_state = twin_response.json()
    return battery_state, new_soc
```

##### 4. Data Logging
```python
def log_simulation_data(step, soc, current, power, battery_state):
    """Logs comprehensive simulation data"""
    data_point = {
        'step': step,
        'timestamp': time.time(),
        'soc': soc,
        'pack_current_A': current,
        'power_kw': power,
        'soh': battery_state['soh'],
        'pack_temp_C': battery_state['pack_temp_C'],
        'internal_resistance_ohm': battery_state['r_int_ohm'],
        'max_discharge_kW': battery_state['max_discharge_kW'],
        'ambient_temp_C': ambient_temp,
        'motor_efficiency': calculate_efficiency(power, current)
    }
    return data_point
```

---

### 🧪 Quick Test Script

#### `beamng_client/kia_working_test.py`

**Purpose**: Rapid validation that Kia EV3 integration works

```python
def test_kia_ev3():
    """Quick 200-step test of Kia EV3 battery simulation"""
    
    # 1. Connect to BeamNG
    beamng = BeamNGpy('localhost', 64256, home=CFG['beamng_home'])
    beamng.open()
    
    # 2. Create Kia EV3 vehicle
    vehicle = Vehicle('ev', model='sv1ev3', license='TWIN01')
    
    # 3. Set up scenario
    scenario = Scenario('smallgrid', f'kia_test_{random.randint(1000,9999)}')
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))
    scenario.make(beamng)
    beamng.load_scenario(scenario)
    beamng.start_scenario()
    
    # 4. Run test simulation
    soc = 0.8
    for step in range(200):
        # Simple power cycling pattern
        if step % 40 < 10:
            power_kw = 80.0      # High power
        elif step % 40 < 20:
            power_kw = 35.0      # Medium power  
        elif step % 40 < 30:
            power_kw = -60.0     # Regeneration
        else:
            power_kw = 15.0      # Low power
        
        # Calculate and log results
        current = est_pack_current(power_kw)
        # ... battery twin call and logging
```

---

### 🌐 Battery Digital Twin Service

#### `twin_service/api.py`

**FastAPI REST Service for Battery Physics**

##### 1. API Endpoint
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Battery Digital Twin", version="1.0.0")

class BatteryRequest(BaseModel):
    pack_current_A: float      # Battery current in amperes
    soc: float                 # State of charge (0.0-1.0)
    amb_temp_C: float          # Ambient temperature in Celsius
    dt_s: float                # Time step in seconds

@app.post("/step")
async def battery_step(request: BatteryRequest):
    """Process one battery simulation timestep"""
    
    # Update battery state based on inputs
    result = battery_model.step(
        current=request.pack_current_A,
        soc=request.soc,
        ambient_temp=request.amb_temp_C,
        dt=request.dt_s
    )
    
    return {
        "soh": result.state_of_health,
        "cap_Ah": result.capacity_ah,
        "r_int_ohm": result.internal_resistance,
        "pack_temp_C": result.pack_temperature,
        "max_discharge_kW": result.max_power_discharge,
        "max_regen_kW": result.max_power_regen
    }
```

##### 2. Battery Physics Model
```python
class BatteryModel:
    def __init__(self):
        self.soh = 1.0                    # State of health
        self.capacity_ah = 220.0          # Amp-hour capacity
        self.internal_resistance = 0.1    # Ohms
        self.pack_temperature = 25.0      # Celsius
        
    def step(self, current, soc, ambient_temp, dt):
        """Update battery state for one timestep"""
        
        # 1. Thermal model
        self.update_temperature(current, ambient_temp, dt)
        
        # 2. Degradation model  
        self.update_degradation(current, soc, self.pack_temperature, dt)
        
        # 3. Update capacity and resistance
        self.capacity_ah = 220.0 * self.soh
        self.internal_resistance = 0.1 / self.soh
        
        # 4. Calculate power limits
        max_discharge = self.calculate_max_power(soc, self.pack_temperature)
        max_regen = self.calculate_max_regen(soc, self.pack_temperature)
        
        return BatteryState(
            state_of_health=self.soh,
            capacity_ah=self.capacity_ah,
            internal_resistance=self.internal_resistance,
            pack_temperature=self.pack_temperature,
            max_power_discharge=max_discharge,
            max_power_regen=max_regen
        )
    
    def update_degradation(self, current, soc, temp, dt):
        """Calculate battery degradation"""
        
        # Cycle degradation (based on current magnitude)
        cycle_stress = abs(current) / 400.0  # Normalize to max current
        
        # Thermal degradation (Arrhenius equation)
        thermal_factor = exp(-5000 / (8.314 * (temp + 273.15)))
        
        # SOC stress (higher degradation at extreme SOCs)
        if soc < 0.2 or soc > 0.9:
            soc_stress = 2.0
        else:
            soc_stress = 1.0
            
        # Combined degradation rate (per second)
        degradation_rate = 1e-8 * cycle_stress * thermal_factor * soc_stress
        
        # Update SOH
        self.soh -= degradation_rate * dt
        self.soh = max(0.7, self.soh)  # Minimum 70% health
```

---

### 📊 Original Basic Simulation

#### `beamng_client/stream.py`

**Legacy simulation script for basic vehicles**

```python
def run():
    """Basic EV simulation with fallback vehicles"""
    
    # Try multiple vehicle models
    ev_models = ['etk800', 'vivace', 'sunburst', 'coupe']
    
    vehicle = None
    for model in ev_models:
        try:
            vehicle = Vehicle('ev', model=model)
            print(f"✅ Using vehicle model: {model}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model}: {e}")
            continue
    
    # Basic simulation loop with hardcoded values
    soc = 0.8
    for step in range(20000):
        # Simple power calculation
        wheel_torque = 100.0  # Nm
        shaft_rps = 150.0     # RPM
        mech_power_kw = (wheel_torque * shaft_rps) / 1000.0
        
        # Call battery twin
        current = est_pack_current(mech_power_kw)
        twin_response = twin_step(current, soc, 30.0, 0.05)
        
        # Update SOC
        soc -= (current * 0.05) / (220.0 * 3600.0)
        
        if step % 50 == 0:
            print(f"SOC={soc:.3f} Power={mech_power_kw:.1f}kW")
```

---

## 🔋 Battery Degradation Detection

### How the System Tracks Battery Health

#### 1. **State of Health (SOH) Monitoring**
```python
# Tracks capacity retention over time
initial_soh = 1.000000
current_soh = 0.999999999768  # After 2000 simulation steps
degradation = initial_soh - current_soh  # 2.32e-10
```

#### 2. **Degradation Mechanisms**
- **Cycling Stress**: High charge/discharge currents
- **Thermal Stress**: Operation outside optimal temperature range
- **Calendar Aging**: Time-based degradation
- **SOC Stress**: Extreme state of charge levels

#### 3. **Physical Parameter Changes**
```python
# As battery degrades:
capacity_ah = original_capacity * soh         # Capacity loss
resistance_ohm = original_resistance / soh    # Resistance increase
max_power = calculate_power_limits(soh, temp) # Power capability reduction
```

---

## 🚀 How to Run the System

### 1. **Setup Environment**
```bash
# Create virtual environment
python -m venv env
env\Scripts\activate

# Install dependencies
pip install beamngpy fastapi uvicorn requests pyyaml numpy
```

### 2. **Configure BeamNG Path**
Edit `beamng_client/config.yaml`:
```yaml
beamng_home: "C:/YourPath/BeamNG.tech"
```

### 3. **Start Battery Twin Service**
```bash
cd twin_service
uvicorn api:app --host 127.0.0.1 --port 8008
```

### 4. **Run Kia EV3 Simulation**
```bash
# Quick test (200 steps)
python beamng_client/kia_working_test.py

# Full simulation (2000 steps)
python beamng_client/kia_ev3_simulation.py
```

---

## 📈 Sample Results Analysis

### From Real Test Run: `kia_ev3_results_aggressive_1754665164.json`

#### **Battery Performance**
- **Energy Consumed**: 5.4 kWh over 2000 steps
- **SOC Change**: 80.0% → 73.4% (6.6% usage)
- **Max Current**: 443.3A discharge, 391.7A regen
- **Power Range**: -150kW (regen) to +150kW (discharge)

#### **Thermal Behavior**
- **Pack Temperature**: 24.9°C → 25.3°C (0.4°C rise)
- **Ambient Variation**: 20.8°C → 27.9°C (programmed variation)
- **Thermal Management**: Stable pack temperature despite high power

#### **Degradation Tracking**
- **SOH Loss**: 2.32 × 10⁻¹⁰ (minimal for short test)
- **Capacity**: Maintained ~220Ah throughout test
- **Resistance**: Stable internal resistance

---

## 🎯 System Capabilities

### **Real-Time Battery Simulation**
- ✅ Accurate current/power calculations
- ✅ Thermal modeling with ambient effects
- ✅ Degradation tracking over time
- ✅ SOC management with coulomb counting

### **Realistic Vehicle Integration**
- ✅ Kia EV3 specifications (81.4kWh battery)
- ✅ Multiple driving patterns (city/highway/aggressive)
- ✅ AI-driven vehicle behavior
- ✅ Regenerative braking simulation

### **Comprehensive Data Logging**
- ✅ JSON export of all simulation data
- ✅ Real-time monitoring and visualization
- ✅ Battery health trend analysis
- ✅ Performance optimization insights

### **Modular Architecture**
- ✅ Separate battery physics service
- ✅ Configurable vehicle models
- ✅ Extensible driving patterns
- ✅ REST API for integration

---

## 🔧 Troubleshooting

### **Common Issues**

#### BeamNG Connection Failed
```bash
# Check BeamNG path in config.yaml
# Ensure BeamNG.tech is installed, not BeamNG.drive
# Verify port 64256 is available
```

#### Vehicle Model Not Found
```bash
# Run vehicle availability test
python beamng_client/test_vehicles.py

# Use fallback model 'etk800' if Kia EV3 not available
```

#### Battery Twin Service Error
```bash
# Check if service is running
curl http://127.0.0.1:8008/docs

# Restart service if needed
uvicorn twin_service.api:app --reload
```

---

## 🎓 Educational Value

This project demonstrates:

### **Electric Vehicle Systems**
- Battery pack physics and thermal management
- Motor efficiency and power electronics
- Regenerative braking energy recovery
- State estimation and battery management

### **Digital Twin Technology**
- Real-time physics simulation
- Model validation with virtual testing
- Predictive maintenance applications
- IoT integration patterns

### **Software Engineering**
- Microservices architecture
- REST API design and integration
- Real-time data streaming
- Configuration management

### **Data Science Applications**
- Time series analysis of battery data
- Degradation pattern recognition
- Performance optimization
- Predictive modeling

---

## 📚 References & Standards

- **Battery Testing**: IEC 62660 (Li-ion battery testing)
- **EV Standards**: ISO 15118 (Vehicle-to-Grid communication)
- **Thermal Modeling**: Based on electrochemical-thermal models
- **Degradation**: Calendar and cycle aging mechanisms from academic literature

---

## 🏆 Project Achievements

✅ **Realistic EV Simulation**: Authentic Kia EV3 behavior  
✅ **Battery Digital Twin**: Physics-based battery modeling  
✅ **Real-Time Operation**: 50ms timestep simulation  
✅ **Comprehensive Logging**: Detailed performance data  
✅ **Degradation Tracking**: Health monitoring over time  
✅ **Modular Design**: Extensible and maintainable code  

This system provides a sophisticated platform for electric vehicle battery research, validation, and optimization using industry-standard simulation tools and modern software architecture.