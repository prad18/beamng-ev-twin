"""
Battery Degradation ML Model Training Pipeline
===============================================
Train LSTM, Random Forest, and Physics-Informed models for battery SOH prediction.

This integrates:
1. External datasets (NASA, Stanford/MIT)
2. BeamNG simulation data
3. PyBaMM physics model
4. Ensemble prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Tuple, Dict, List

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML libraries not installed. Run: pip install scikit-learn tensorflow")
    ML_AVAILABLE = False


class BatteryDegradationMLPipeline:
    """
    Complete ML pipeline for battery degradation prediction.
    """
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def load_external_datasets(self) -> pd.DataFrame:
        """Load and combine external datasets."""
        print("📊 Loading external datasets...")
        
        datasets = []
        
        # Load Stanford/MIT dataset if available
        stanford_path = self.data_dir / "stanford_mit" / "processed_features.csv"
        if stanford_path.exists():
            stanford_df = pd.read_csv(stanford_path)
            stanford_df['source'] = 'stanford'
            datasets.append(stanford_df)
            print(f"   ✅ Loaded Stanford/MIT: {len(stanford_df)} samples")
        
        # Load NASA dataset if available
        nasa_path = self.data_dir / "nasa" / "processed_features.csv"
        if nasa_path.exists():
            nasa_df = pd.read_csv(nasa_path)
            nasa_df['source'] = 'nasa'
            datasets.append(nasa_df)
            print(f"   ✅ Loaded NASA: {len(nasa_df)} samples")
        
        if not datasets:
            print("   ⚠️ No external datasets found. Using synthetic data.")
            return self._generate_synthetic_data()
        
        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"\n   📊 Total samples: {len(combined_df)}")
        
        return combined_df
    
    def load_beamng_data(self) -> pd.DataFrame:
        """Load BeamNG simulation results."""
        print("\n📊 Loading BeamNG simulation data...")
        
        # Find all BeamNG result files
        result_files = list(Path("../").glob("kia_ev3_results_*.json"))
        
        if not result_files:
            print("   ⚠️ No BeamNG results found")
            return pd.DataFrame()
        
        all_data = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                df['source'] = 'beamng'
                all_data.append(df)
                print(f"   ✅ Loaded {result_file.name}: {len(df)} timesteps")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {result_file.name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n   📊 Total BeamNG samples: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic battery degradation data for testing."""
        print("   🔄 Generating synthetic data...")
        
        np.random.seed(42)
        
        # Simulate battery cycles with degradation
        data = []
        
        for i in range(n_samples):
            # Random battery parameters
            initial_capacity = 220.0
            c_rate = np.random.uniform(0.5, 2.0)
            temperature = np.random.uniform(15, 40)
            soc_mean = np.random.uniform(0.3, 0.8)
            
            # Degradation model (simplified)
            base_degradation = 0.0001
            temp_factor = np.exp((temperature - 25) / 15)
            c_rate_factor = c_rate ** 0.5
            
            degradation_rate = base_degradation * temp_factor * c_rate_factor
            
            # Simulate cycles
            cycles = np.random.randint(100, 2000)
            soh = 1.0 - (degradation_rate * cycles)
            soh = max(0.7, soh)  # Min 70% SOH
            
            capacity = initial_capacity * soh
            
            data.append({
                'c_rate': c_rate,
                'temperature': temperature,
                'soc_mean': soc_mean,
                'cycles': cycles,
                'capacity': capacity,
                'soh': soh,
                'source': 'synthetic'
            })
        
        df = pd.DataFrame(data)
        print(f"   ✅ Generated {len(df)} synthetic samples")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Engineer features from raw data.
        
        Returns:
            X: Feature matrix
            y: Target values (SOH)
            feature_names: List of feature names
        """
        print("\n🔧 Engineering features...")
        
        # Ensure we have required columns
        if 'soh' not in df.columns and 'capacity' in df.columns:
            # Calculate SOH from capacity
            df['soh'] = df['capacity'] / 220.0  # Assuming 220Ah nominal
        
        # Basic features that should exist in most datasets
        feature_columns = []
        
        # Cycling features
        if 'cycles' in df.columns:
            feature_columns.append('cycles')
        
        if 'c_rate' in df.columns:
            feature_columns.append('c_rate')
        elif 'current_A' in df.columns and 'capacity' in df.columns:
            df['c_rate'] = df['current_A'].abs() / df['capacity']
            feature_columns.append('c_rate')
        
        # Temperature features
        if 'temperature' in df.columns:
            feature_columns.append('temperature')
        elif 'pack_temp_C' in df.columns:
            df['temperature'] = df['pack_temp_C']
            feature_columns.append('temperature')
        
        if 'ambient_temp_C' in df.columns:
            feature_columns.append('ambient_temp_C')
        
        # SOC features
        if 'soc_mean' in df.columns:
            feature_columns.append('soc_mean')
        elif 'soc' in df.columns:
            feature_columns.append('soc')
        
        # Power features (if available)
        if 'power_kw' in df.columns:
            feature_columns.append('power_kw')
            # Energy throughput
            df['energy_throughput'] = df['power_kw'].cumsum() * 0.05 / 3600  # Assuming 50ms timestep
            feature_columns.append('energy_throughput')
        
        # Derived features
        if 'temperature' in feature_columns:
            # Temperature stress
            df['temp_stress'] = (df['temperature'] - 25).abs()
            feature_columns.append('temp_stress')
        
        if 'soc' in df.columns or 'soc_mean' in df.columns:
            soc_col = 'soc' if 'soc' in df.columns else 'soc_mean'
            # SOC stress (high/low SOC increases degradation)
            df['soc_stress'] = ((df[soc_col] - 0.5).abs() > 0.3).astype(float)
            feature_columns.append('soc_stress')
        
        # Ensure we have features
        if not feature_columns:
            raise ValueError("No valid features found in dataset!")
        
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Extract features and target
        X = df[feature_columns].fillna(0).values
        y = df['soh'].fillna(1.0).values
        
        print(f"   ✅ Engineered {len(feature_columns)} features")
        print(f"   Features: {feature_columns}")
        print(f"   Samples: {len(X)}")
        
        self.feature_names = feature_columns
        
        return X, y, feature_columns
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model."""
        print("\n🌲 Training Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        print("   ✅ Random Forest trained")
        
        return rf_model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor:
        """Train Gradient Boosting model."""
        print("\n📈 Training Gradient Boosting...")
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        
        print("   ✅ Gradient Boosting trained")
        
        return gb_model
    
    def build_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build LSTM model for sequence prediction."""
        print("\n🧠 Building LSTM model...")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # SOH between 0 and 1
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("   ✅ LSTM model built")
        model.summary()
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> keras.Model:
        """Train LSTM model."""
        print("\n🧠 Training LSTM...")
        
        # Reshape for LSTM (samples, timesteps, features)
        # For now, treat each sample as single timestep
        X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        model = self.build_lstm_model((1, X_train.shape[1]))
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("   ✅ LSTM trained")
        
        return model
    
    def evaluate_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate all models."""
        print("\n📊 Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            if name == 'lstm':
                X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                y_pred = model.predict(X_test_seq, verbose=0).flatten()
            else:
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\n   {name.upper()}:")
            print(f"      MAE:  {mae:.4f}")
            print(f"      RMSE: {rmse:.4f}")
            print(f"      R²:   {r2:.4f}")
        
        return results
    
    def save_models(self, models: Dict, output_dir: str = "ml_models/trained"):
        """Save trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving models to {output_path}...")
        
        for name, model in models.items():
            if name == 'lstm':
                model_path = output_path / f"{name}_model.h5"
                model.save(model_path)
            else:
                model_path = output_path / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"   ✅ Saved {name}")
        
        # Save scaler
        scaler_path = output_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers.get('standard'), f)
        
        # Save feature names
        features_path = output_path / "features.json"
        with open(features_path, 'w') as f:
            json.dump({'features': self.feature_names}, f, indent=2)
        
        print("   ✅ All models saved")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("🚀 Starting Battery Degradation ML Pipeline")
        print("="*60)
        
        # Load data
        external_data = self.load_external_datasets()
        beamng_data = self.load_beamng_data()
        
        # Combine all data
        if not beamng_data.empty:
            all_data = pd.concat([external_data, beamng_data], ignore_index=True)
        else:
            all_data = external_data
        
        print(f"\n📊 Total dataset size: {len(all_data)}")
        
        # Engineer features
        X, y, feature_names = self.engineer_features(all_data)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['standard'] = scaler
        
        print(f"\n📊 Feature scaling complete")
        print(f"   Mean: {scaler.mean_}")
        print(f"   Std:  {scaler.scale_}")
        
        # Train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training:   {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test:       {len(X_test)} samples")
        
        # Train models
        models = {}
        
        models['random_forest'] = self.train_random_forest(X_train, y_train)
        models['gradient_boosting'] = self.train_gradient_boosting(X_train, y_train)
        
        if ML_AVAILABLE and len(X_train) > 100:
            models['lstm'] = self.train_lstm(X_train, y_train, X_val, y_val)
        
        # Evaluate
        results = self.evaluate_models(models, X_test, y_test)
        
        # Save models
        self.save_models(models)
        
        # Save results
        results_path = Path("ml_models/trained/evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ Training pipeline complete!")
        print(f"\n📊 Best model: {min(results.items(), key=lambda x: x[1]['mae'])[0]}")
        
        return models, results


if __name__ == "__main__":
    if not ML_AVAILABLE:
        print("❌ ML libraries not available!")
        print("Install: pip install scikit-learn tensorflow pandas numpy")
        exit(1)
    
    # Run pipeline
    pipeline = BatteryDegradationMLPipeline()
    models, results = pipeline.run_full_pipeline()
    
    print("\n📚 Next steps:")
    print("   1. Integrate with PyBaMM: Update api_pybamm.py")
    print("   2. Deploy models: Copy to twin_service/")
    print("   3. Test predictions: python ml_models/test_predictions.py")
