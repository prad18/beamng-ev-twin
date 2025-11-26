# 🗄️ Battery Degradation Datasets for ML Training
# ==================================================

"""
This document lists curated, publicly available battery degradation datasets
that can be used to train machine learning models for the EV Battery Digital Twin project.
"""

## 📊 RECOMMENDED DATASETS (High Quality & Well-Documented)

### 1. **NASA Battery Dataset** ⭐⭐⭐⭐⭐
**Source**: NASA Prognostics Center of Excellence (PCoE)
**URL**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

**Description**:
- 28 commercial Li-ion batteries (18650 cells)
- Multiple charge/discharge cycles at different temperatures
- Capacity fade measurements over full battery lifetime
- Impedance measurements at regular intervals

**Data Features**:
- Voltage, Current, Temperature, Time
- Capacity measurements
- Internal resistance
- Full cycling until end-of-life (70-80% capacity)

**Size**: ~2 GB
**Format**: MATLAB (.mat) files, CSV available

**Use Case for Your Project**:
- Train degradation prediction models (SOH estimation)
- Validate PyBaMM model accuracy
- Develop capacity fade forecasting algorithms

**Download Script**:
```python
# datasets/download_nasa.py
import requests
import os

def download_nasa_battery_data():
    """Download NASA battery aging dataset."""
    base_url = "https://ti.arc.nasa.gov/c/6/"
    
    batteries = [
        "B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat",
        # Add more as needed
    ]
    
    os.makedirs("datasets/nasa", exist_ok=True)
    
    for battery in batteries:
        url = f"{base_url}{battery}"
        print(f"Downloading {battery}...")
        response = requests.get(url)
        
        with open(f"datasets/nasa/{battery}", 'wb') as f:
            f.write(response.content)
    
    print("✅ NASA dataset downloaded!")
```

---

### 2. **Stanford/MIT Battery Cycling Dataset** ⭐⭐⭐⭐⭐
**Source**: Stanford Energy, MIT Data-Driven Battery Design
**URL**: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

**Description**:
- 124 commercial LFP/graphite cells (A123 APR18650M1A)
- Fast-charging optimization study
- Extensive cycling data with various charging protocols
- Early-life predictions of battery lifetime

**Data Features**:
- High-resolution voltage/current/temperature profiles
- Discharge capacity measurements
- Cycle life data (up to 2000+ cycles)
- Early degradation indicators

**Size**: ~10 GB (complete dataset)
**Format**: Python pickle (.pkl), CSV available

**Use Case**:
- Train early-life prediction models
- Fast-charging impact on degradation
- Cycle-to-cycle degradation patterns
- Validate PyBaMM predictions

**Citation**:
```
Severson et al., "Data-driven prediction of battery cycle life before capacity degradation"
Nature Energy, 2019. DOI: 10.1038/s41560-019-0356-8
```

---

### 3. **Oxford Battery Degradation Dataset** ⭐⭐⭐⭐
**Source**: University of Oxford, Department of Engineering Science
**URL**: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac

**Description**:
- 8 commercial Li-ion cells
- Drive cycle aging (realistic EV usage patterns)
- Different temperatures and C-rates
- Comprehensive electrochemical impedance spectroscopy (EIS)

**Data Features**:
- Real-world drive cycle data
- Temperature variation effects
- Impedance growth tracking
- Capacity and resistance measurements

**Size**: ~500 MB
**Format**: MATLAB, CSV

**Use Case**:
- Train models on realistic driving patterns (similar to BeamNG data)
- Temperature-dependent degradation
- Impedance-based SOH estimation

---

### 4. **CALCE Battery Dataset** ⭐⭐⭐⭐
**Source**: University of Maryland, Center for Advanced Life Cycle Engineering
**URL**: https://web.calce.umd.edu/batteries/data.htm

**Description**:
- Multiple Li-ion chemistries (LCO, NMC, LFP)
- Various cell formats (18650, pouch, prismatic)
- Different operating conditions
- Long-term cycling and calendar aging

**Data Features**:
- Voltage, current, temperature time series
- Capacity measurements
- Calendar aging data (storage conditions)
- Cycle life data

**Size**: ~1-2 GB per dataset
**Format**: Excel, MATLAB

**Use Case**:
- Calendar aging models
- Multi-chemistry comparison
- Storage condition impact on degradation

---

### 5. **Toyota Research Institute (TRI) Dataset** ⭐⭐⭐⭐⭐
**Source**: Toyota Research Institute
**URL**: https://data.matr.io/1/projects/5d80e633f405260001c0b60a

**Description**:
- 48 commercial LFP cells
- Long-term degradation study
- Multiple temperature and C-rate combinations
- Extensive cycling (up to 1000+ cycles)

**Data Features**:
- High-quality voltage/current curves
- Differential capacity analysis
- Temperature-controlled aging
- Comprehensive metadata

**Size**: ~5 GB
**Format**: HDF5, CSV

**Use Case**:
- Temperature impact modeling
- LFP-specific degradation (matches Kia EV3)
- Long-term SOH forecasting

---

## 🔧 DATASET PREPROCESSING PIPELINE

### Create Data Loader (`ml_models/data_loader.py`)

```python
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, Dict

class BatteryDataLoader:
    """Universal battery dataset loader and preprocessor."""
    
    def __init__(self, dataset_path: str, dataset_type: str = "nasa"):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        
    def load_nasa(self) -> pd.DataFrame:
        """Load NASA battery dataset."""
        mat_data = loadmat(self.dataset_path)
        
        # Extract cycle data
        cycles = []
        battery_name = self.dataset_path.stem
        
        for i, cycle in enumerate(mat_data[battery_name]['cycle'][0][0][0]):
            cycles.append({
                'cycle': i,
                'voltage': cycle['data'][0][0]['Voltage_measured'][0][0].flatten(),
                'current': cycle['data'][0][0]['Current_measured'][0][0].flatten(),
                'temperature': cycle['data'][0][0]['Temperature_measured'][0][0].flatten(),
                'capacity': cycle['data'][0][0]['Capacity'][0][0][0][0],
                'time': cycle['data'][0][0]['Time'][0][0].flatten()
            })
        
        return pd.DataFrame(cycles)
    
    def load_stanford(self) -> pd.DataFrame:
        """Load Stanford/MIT fast-charging dataset."""
        # Typically stored in pickle format
        import pickle
        
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame
        processed_data = []
        for cell_id, cell_data in data.items():
            for cycle_idx, cycle in enumerate(cell_data['cycles']):
                processed_data.append({
                    'cell_id': cell_id,
                    'cycle': cycle_idx,
                    'capacity': cycle['Qd'],
                    'voltage': cycle['V'],
                    'current': cycle['I'],
                    'temperature': cycle['T']
                })
        
        return pd.DataFrame(processed_data)
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ML features from battery cycles.
        
        Returns:
            X: Feature matrix (cycle-level features)
            y: Target values (capacity fade)
        """
        
        features = []
        targets = []
        
        for cycle_idx in df['cycle'].unique():
            cycle_data = df[df['cycle'] == cycle_idx]
            
            # Time-series features
            voltage = cycle_data['voltage'].values
            current = cycle_data['current'].values
            
            # Statistical features
            feature_dict = {
                # Voltage features
                'v_mean': np.mean(voltage),
                'v_std': np.std(voltage),
                'v_max': np.max(voltage),
                'v_min': np.min(voltage),
                'v_range': np.max(voltage) - np.min(voltage),
                
                # Current features
                'i_mean': np.mean(current),
                'i_max': np.max(current),
                
                # Cycle characteristics
                'cycle_time': cycle_data['time'].max(),
                'energy_throughput': np.trapz(voltage * current, cycle_data['time']),
                
                # Degradation indicators
                'dv_dt': np.gradient(voltage).mean(),
                'internal_resistance_est': (voltage.max() - voltage.min()) / current.max()
            }
            
            features.append(list(feature_dict.values()))
            targets.append(cycle_data['capacity'].iloc[0])
        
        return np.array(features), np.array(targets)
```

---

## 🤖 INTEGRATION WITH YOUR PROJECT

### Step 1: Download and Organize Datasets

```bash
# Create dataset directory
mkdir -p datasets/{nasa,stanford,oxford,calce,tri}

# Download datasets
python datasets/download_nasa.py
python datasets/download_stanford.py
# etc.
```

### Step 2: Preprocess for ML Training

Create `ml_models/preprocess_datasets.py`:

```python
from data_loader import BatteryDataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Load multiple datasets
nasa_loader = BatteryDataLoader("datasets/nasa/B0005.mat", "nasa")
nasa_df = nasa_loader.load_nasa()

# Extract features
X_nasa, y_nasa = nasa_loader.extract_features(nasa_df)

# Combine with BeamNG simulation data
beamng_data = pd.read_json("kia_ev3_results_aggressive_1754665164.json")

# Feature alignment
# Ensure BeamNG features match dataset features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_nasa, y_nasa, test_size=0.2, random_state=42
)

# Save processed data
np.save("ml_models/data/X_train.npy", X_train)
np.save("ml_models/data/X_test.npy", X_test)
np.save("ml_models/data/y_train.npy", y_train)
np.save("ml_models/data/y_test.npy", y_test)
```

### Step 3: Train ML Models

Create `ml_models/train_degradation_model.py`:

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load processed data
X_train = np.load("ml_models/data/X_train.npy")
X_test = np.load("ml_models/data/X_test.npy")
y_train = np.load("ml_models/data/y_train.npy")
y_test = np.load("ml_models/data/y_test.npy")

# LSTM Model for sequence prediction
def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # SOH prediction
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Random Forest for baseline
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
rf_score = rf_model.score(X_test, y_test)
print(f"Random Forest R²: {rf_score:.4f}")
```

---

## 📈 DATASET COMPARISON TABLE

| Dataset | Size | Cells | Cycles | Chemistry | Best For |
|---------|------|-------|--------|-----------|----------|
| NASA | 2 GB | 28 | 100-200 | Li-ion | General degradation |
| Stanford/MIT | 10 GB | 124 | 2000+ | LFP | Fast charging, early prediction |
| Oxford | 500 MB | 8 | Varies | Li-ion | Drive cycles, EIS |
| CALCE | 2 GB | Multiple | 1000+ | Various | Calendar aging, multi-chemistry |
| TRI | 5 GB | 48 | 1000+ | LFP | Temperature effects, LFP-specific |

---

## 🎯 RECOMMENDED APPROACH FOR YOUR PROJECT

### Phase 1: Data Collection (Week 1-2)
1. Download NASA dataset (lightweight, well-documented)
2. Download Stanford/MIT dataset (most comprehensive for prediction)
3. Collect BeamNG simulation data using your existing scripts

### Phase 2: Feature Engineering (Week 2-3)
1. Align BeamNG features with dataset features
2. Create unified preprocessing pipeline
3. Generate synthetic training data using PyBaMM

### Phase 3: Model Training (Week 3-4)
1. Train baseline Random Forest model
2. Train LSTM for sequence prediction
3. Train Physics-Informed Neural Network (PINN) combining PyBaMM

### Phase 4: Integration (Week 4-5)
1. Deploy trained model alongside PyBaMM service
2. Create ensemble prediction (ML + Physics)
3. Real-time inference in dashboard

---

## 🔗 ADDITIONAL RESOURCES

### Papers with Datasets:
1. **Severson et al. (2019)**: "Data-driven prediction of battery cycle life"
2. **Krewer et al. (2018)**: "Review—Dynamic Models of Li-Ion Batteries"
3. **Roman et al. (2021)**: "Machine learning pipeline for battery state-of-health"

### Tools:
- **BatteryArchive.org**: Centralized battery data repository
- **BEEP (Battery Evaluation and Early Prediction)**: Stanford data processing pipeline
- **PyBaMM-dataverse**: Integrated datasets for PyBaMM validation

---

## ✅ NEXT STEPS

1. **Run**: `python datasets/download_nasa.py` to get started
2. **Process**: Use `data_loader.py` to extract features
3. **Train**: Run `train_degradation_model.py` for baseline model
4. **Integrate**: Add ML predictions to your dashboard

This will give you a production-ready battery degradation prediction system! 🚀
"""

# Save this as a markdown file
if __name__ == "__main__":
    print(__doc__)
