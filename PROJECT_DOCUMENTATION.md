# 🔋 BeamNG EV Battery Digital Twin Project Documentation

## 📋 Project Overview

We're building a **comprehensive battery degradation prediction system** that combines realistic vehicle simulation with advanced machine learning to understand how electric vehicle batteries age under real-world driving conditions.

### 🎯 **What We're Building**
- **Real-time battery simulation** using actual vehicle data from BeamNG.tech
- **Machine learning models** to predict battery degradation patterns
- **Digital twin integration** with professional simulation tools (AMESIM/Simulink)
- **Comprehensive testing platform** for battery management optimization

### 🚀 **Why This Matters**
- Traditional battery testing takes **months/years** - we can simulate this in **hours**
- Real driving patterns cause different degradation than lab tests
- Predictive maintenance can save **millions** in battery replacement costs
- Optimal charging strategies can extend battery life by **20-30%**

---

## 🏗️ Current System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BeamNG.tech   │───▶│  Python Scripts │───▶│ Battery Twin    │
│  (Vehicle Sim)  │    │  (Data Extract) │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Realistic Drive │    │ Motor/Battery   │    │ SOC/SOH/Temp    │
│   Patterns      │    │     Data        │    │   Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🚗 Phase 1: Vehicle Simulation Foundation (COMPLETED ✅)

### **What We Did**
We connected BeamNG.tech (a realistic vehicle physics simulator) to extract real motor and battery data from electric vehicles.

### **How It Works**
```
BeamNG.tech Simulator → Python Scripts → Battery Data Extraction → JSON Files
```

### **Key Achievements**

#### 1. **Vehicle Integration**
- ✅ Successfully connected to **Kia EV3** (`sv1ev3`) 
- ✅ Successfully connected to **Tesla Model 3** (`Model3_2024`)
- ✅ Created universal vehicle data exposer for any EV model

#### 2. **Real Motor Data Extraction**
- ✅ **Motor Torque**: Real-time torque output (Nm)
- ✅ **Motor Speed**: Angular velocity (rad/s → RPM conversion)
- ✅ **Power Calculation**: `Power (kW) = (Torque × Speed) / 1000`
- ✅ **Regenerative Braking**: Automatic detection of regen events

#### 3. **Battery-Specific Sensors**
- ✅ **Electrics Sensor**: Basic vehicle electrical data
- ✅ **PowertrainSensor**: Detailed motor and drivetrain data
- ✅ **Lua Queries**: Deep vehicle system access for battery parameters

#### 4. **Data Collection System**
- ✅ **Organized file storage**: `battery_sensor_data/` folder
- ✅ **Timestamped results**: Each test saves with datetime
- ✅ **Multiple data formats**: Separate files for electrics, powertrain, analysis
- ✅ **Automated analysis**: Built-in data quality assessment

### **Sample Data We're Getting**

#### **Motor Data (PowertrainSensor)**
```json
"rearMotor": {
  "outputTorque1": -2.0000573632709,  // Nm (negative = regen!)
  "outputAV1": 0.11472654178739004,   // rad/s (motor speed)
  "inputAV": 0.0                      // Motor input speed
}
```

#### **Regenerative Braking Data**
```json
"regenThrottle": 0.0,        // Current regen pedal (0-1)
"regenStrength": 3.0,        // Regen intensity (1-5)
"regenFromOnePedal": 0.0,    // One-pedal driving regen
"regenFromBrake": 0.0,       // Brake pedal regen
"maxRegenStrength": 3.0      // Max regen available
```

### **Why This Data is Revolutionary**
- **Real driving patterns** instead of synthetic test cycles
- **Actual motor behavior** under realistic loads
- **Dynamic power fluctuations** during acceleration, cruising, braking
- **Regenerative braking events** automatically captured

---

## ⚡ Phase 2: Battery Physics Engine (IN PROGRESS 🔄)

### **What We're Building**
A physics-based battery simulation that uses real motor data to calculate realistic degradation.

### **Current Implementation**

#### **FastAPI Battery Twin Service**
- 🔄 **REST API**: `POST /step` endpoint for battery calculations
- 🔄 **LiFePO4 Model**: 220Ah capacity, realistic internal resistance
- 🔄 **SOC Tracking**: State of Charge calculation from current flow
- 🔄 **SOH Monitoring**: State of Health degradation over time
- 🔄 **Thermal Model**: Temperature-dependent behavior

#### **Real Power Integration**
```python
# Instead of estimated power...
power_kw = 80.0  # Fixed estimate ❌

# We now use REAL BeamNG motor data ✅
motor_torque = powertrain_data['rearMotor']['outputTorque1']  
motor_speed = powertrain_data['rearMotor']['outputAV1']
real_power_kw = (motor_torque * motor_speed) / 1000

# Handle regenerative braking automatically
if real_power_kw < 0:
    print(f"🔋 Regenerating: {abs(real_power_kw):.2f} kW")
else:
    print(f"⚡ Consuming: {real_power_kw:.2f} kW")
```

### **Battery Degradation Mechanisms**

#### **1. Cycle Degradation**
- **Cause**: Charging and discharging cycles
- **Factors**: C-rate (charge speed), depth of discharge, temperature
- **Effect**: Capacity loss over time

#### **2. Calendar Aging**
- **Cause**: Time-based chemical reactions
- **Factors**: Temperature, state of charge, storage conditions
- **Effect**: Gradual capacity and power fade

#### **3. Thermal Stress**
- **Cause**: High temperatures during operation
- **Factors**: Ambient temperature, cooling system, power demand
- **Effect**: Accelerated aging, safety risks

---

## 🤖 Phase 3: Machine Learning Integration (PLANNED 📋)

### **Why We Need ML**
- **Pattern Recognition**: Identify complex degradation patterns from data
- **Predictive Modeling**: Forecast battery health weeks/months ahead
- **Optimization**: Find optimal charging/driving strategies
- **Anomaly Detection**: Detect unusual battery behavior early

### **Planned ML Models**

#### **1. LSTM Neural Networks**
- **Purpose**: Time series prediction of battery degradation
- **Input**: Historical motor data, SOC, temperature, usage patterns
- **Output**: Future SOH (State of Health) predictions
- **Advantage**: Captures long-term dependencies in battery behavior

#### **2. Physics-Informed Neural Networks (PINNs)**
- **Purpose**: Combine physics laws with ML learning
- **Advantage**: More accurate predictions, needs less training data
- **Implementation**: Neural network that respects battery physics equations

#### **3. Ensemble Models**
- **Purpose**: Combine multiple prediction approaches
- **Models**: LSTM + Random Forest + Physics Model + Transformer
- **Advantage**: More robust predictions, better uncertainty estimation

#### **4. Reinforcement Learning**
- **Purpose**: Optimize charging strategies
- **Goal**: Learn charging patterns that minimize degradation
- **Application**: Smart charging algorithms for fleet management

### **Feature Engineering Plan**

#### **From BeamNG Data**
```python
features = {
    # Power patterns
    'avg_power': 'Average power consumption',
    'max_power': 'Peak power demand',
    'regen_fraction': 'Percentage of time regenerating',
    
    # Usage patterns  
    'soc_swing': 'How much SOC changes per trip',
    'time_high_soc': 'Time spent at high charge levels',
    'fast_charge_events': 'Number of rapid charging events',
    
    # Stress indicators
    'c_rate_99th': '99th percentile charge rate',
    'thermal_stress': 'Heat-related stress accumulation',
    'cycle_depth': 'Average depth of discharge'
}
```

---

## 🔧 Phase 4: Professional Simulation Integration (FUTURE 🚀)

### **Why Professional Tools?**
While our Python/BeamNG system is excellent for development, professional tools offer:
- **Industry validation** (trusted by automotive OEMs)
- **Advanced physics** (electromagnetic, thermal, mechanical coupling)
- **Regulatory compliance** (automotive safety standards)
- **Team collaboration** (standardized modeling practices)

### **AMESIM Integration**

#### **What is AMESIM?**
- **Industry standard** for automotive system simulation
- **Multi-physics modeling**: electrical, thermal, mechanical, hydraulic
- **Battery libraries**: Pre-built EV battery models
- **Real-time capability**: Hardware-in-the-loop testing

#### **How We'll Use It**
```
BeamNG Motor Data → Python Processing → AMESIM Battery Model → Detailed Results
```

#### **Benefits**
- ✅ **Advanced thermal modeling**: 3D heat distribution in battery pack
- ✅ **Cell-level simulation**: Individual cell behavior and balancing
- ✅ **Cooling system modeling**: Liquid/air cooling effectiveness
- ✅ **Safety analysis**: Thermal runaway, overcharge protection

### **MATLAB/Simulink Integration**

#### **What is Simulink?**
- **MathWorks** industry-standard simulation platform
- **Simscape Battery**: Specialized battery modeling toolbox
- **Code generation**: Convert models to C/C++ for real-time systems
- **Extensive libraries**: Control systems, signal processing, ML

#### **How We'll Use It**
```
BeamNG Data → Simulink Model → Battery Management System → Control Optimization
```

#### **Benefits**
- ✅ **Advanced control algorithms**: Model Predictive Control for charging
- ✅ **System-level integration**: Complete vehicle energy management
- ✅ **Hardware deployment**: Generate code for actual battery controllers
- ✅ **Optimization toolbox**: Find optimal operating strategies

### **Digital Twin Architecture**

#### **Complete System Overview**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BeamNG.tech   │    │  Data Pipeline  │    │   AMESIM/       │
│  (Realistic     │───▶│  (Python/ML)    │───▶│   Simulink      │
│   Vehicle)      │    │                 │    │  (Professional) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │  ML Prediction  │              │
         │              │    Engine       │              │
         │              └─────────────────┘              │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   Digital Twin      │
                    │  (Real-time Sync)   │
                    └─────────────────────┘
```

---

## 📊 Data Analysis Results So Far

### **Battery Sensor Analysis Summary**
Based on our Model3_2024 testing:

#### **✅ Data Quality Assessment: EXCELLENT**
- ✅ **Motor data available**: Real torque and speed
- ✅ **Power calculation possible**: Accurate real-time power
- ✅ **Regen events detected**: Automatic regenerative braking capture
- ✅ **Multiple data sources**: Electrics, powertrain, and Lua queries
- ✅ **Degradation testing ready**: All required data for ML training

#### **🔍 What We Discovered**
1. **Tesla Model 3 has rear-wheel drive motor** with detailed torque/speed data
2. **Regenerative braking system** has 5 strength levels (1-5)
3. **Real power calculations** show dynamic behavior impossible to estimate
4. **Comprehensive vehicle data** available for thermal and usage modeling

---

## 🛣️ Project Roadmap

### **Immediate Next Steps (1-2 weeks)**

#### **Phase 2 Completion**
- [ ] **Enhanced battery twin service** with real BeamNG motor data
- [ ] **Time acceleration implementation** for faster degradation testing
- [ ] **Multiple vehicle testing** (compare different EV models)
- [ ] **Comprehensive driving scenarios** (city, highway, aggressive)

#### **ML Foundation (2-4 weeks)**
- [ ] **Data preprocessing pipeline** for ML training
- [ ] **Basic LSTM model** for SOH prediction
- [ ] **Feature engineering** from BeamNG data
- [ ] **Synthetic data generation** for training augmentation

### **Medium Term (1-3 months)**

#### **Advanced ML**
- [ ] **Physics-informed neural networks** implementation
- [ ] **Ensemble model development** (multiple ML approaches)
- [ ] **Uncertainty quantification** for prediction confidence
- [ ] **Real-time prediction service** integration

#### **Professional Tool Integration**
- [ ] **AMESIM model development** for advanced battery physics
- [ ] **Simulink integration** for control system development
- [ ] **Co-simulation setup** between tools
- [ ] **Validation against real-world data**

### **Long Term (3-6 months)**

#### **Production System**
- [ ] **Real-time digital twin** with live vehicle data
- [ ] **Fleet management integration** for multiple vehicles
- [ ] **Predictive maintenance alerts** based on degradation models
- [ ] **Optimization algorithms** for charging and route planning

---

## 💼 Business Applications

### **Automotive Industry**
- **OEM Battery Testing**: Accelerated development cycles
- **Warranty Prediction**: Accurate battery life estimates
- **Design Optimization**: Battery pack and thermal management

### **Fleet Management**
- **Predictive Maintenance**: Schedule battery replacements before failure
- **Route Optimization**: Minimize battery degradation per trip
- **Charging Strategy**: Optimal charging to extend battery life

### **Research & Development**
- **New Battery Chemistry**: Rapid testing of new technologies
- **Climate Impact**: Battery performance in different environments
- **Safety Analysis**: Predict and prevent thermal runaway events

---

## 🎯 Success Metrics

### **Technical Objectives**
- **Prediction Accuracy**: 95%+ SOH prediction accuracy
- **Speed**: 1000x faster than real-time aging tests
- **Coverage**: Support for 10+ different EV models
- **Integration**: Seamless AMESIM/Simulink workflow

### **Business Impact**
- **Cost Reduction**: 50%+ reduction in battery testing time
- **Life Extension**: 20-30% battery life improvement through optimization
- **Safety**: Early detection of potential battery failures
- **Sustainability**: Better battery utilization and recycling timing

---

## 🔧 Technical Requirements

### **Software Stack**
- **Python 3.11+**: Core development language
- **BeamNG.tech**: Vehicle simulation platform
- **FastAPI**: Battery twin service API
- **TensorFlow/PyTorch**: Machine learning frameworks
- **MATLAB/Simulink**: Professional simulation (future)
- **AMESIM**: Multi-physics modeling (future)

### **Hardware Requirements**
- **GPU**: NVIDIA RTX series for ML training
- **RAM**: 32GB+ for large dataset processing
- **Storage**: 1TB+ SSD for simulation data
- **CPU**: Multi-core for parallel simulation

### **Data Management**
- **Real-time data**: Streaming from BeamNG simulation
- **Historical data**: Time-series battery performance
- **Model storage**: Trained ML models and parameters
- **Results archive**: Organized test results and analysis

---

## 👥 Team Collaboration

### **Repository Structure**
```
beamng-ev-twin/
├── beamng_client/          # BeamNG integration scripts
├── twin_service/           # Battery physics service
├── ml_models/              # Machine learning implementations
├── data/                   # Datasets and results
├── documentation/          # Project docs and reports
├── tests/                  # Unit and integration tests
└── deployment/             # Production deployment configs
```

### **Development Workflow**
1. **Feature branches** for new development
2. **Code reviews** for all changes
3. **Automated testing** for reliability
4. **Documentation updates** with each feature
5. **Regular team demos** to show progress

---

## 📚 Learning Resources

### **Battery Technology**
- **Book**: "Battery Management Systems for Large Lithium-Ion Battery Packs" by Davide Andrea
- **Papers**: IEEE Transactions on Vehicular Technology (battery modeling)
- **Courses**: Coursera "Battery Management Systems" by University of Colorado

### **Machine Learning**
- **Platform**: TensorFlow/Keras tutorials for time series
- **Papers**: Physics-Informed Neural Networks research
- **Courses**: Fast.ai for practical ML implementation

### **Simulation Tools**
- **AMESIM**: Siemens official training materials
- **Simulink**: MathWorks Onramp courses
- **BeamNG**: Official BeamNGpy documentation

---

## 🎉 Conclusion

We're building a **cutting-edge battery degradation prediction system** that combines:
- **Realistic vehicle simulation** for authentic data
- **Advanced machine learning** for pattern recognition
- **Professional simulation tools** for industry validation
- **Real-time digital twin** for practical applications

This system will **revolutionize** how we understand and predict battery behavior, leading to longer-lasting, safer, and more efficient electric vehicles.

**The future of EV battery management starts here!** 🚀🔋

---

*Last Updated: September 11, 2025*
*Project Status: Phase 1 Complete ✅, Phase 2 In Progress 🔄*
