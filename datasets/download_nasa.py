"""
NASA Battery Dataset Downloader
================================
Downloads NASA PCoE Li-ion battery aging dataset for ML training.

Dataset Details:
- 28 commercial 18650 Li-ion batteries
- Charge/discharge cycles until end-of-life
- Capacity, voltage, current, temperature, impedance
- Perfect for degradation prediction models
"""

import requests
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def download_file(url: str, destination: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_nasa_battery_dataset():
    """
    Download NASA battery aging dataset.
    
    This downloads the most commonly used batteries from the NASA
    Prognostics Center of Excellence dataset.
    """
    
    print("🔋 NASA Battery Dataset Downloader")
    print("="*60)
    
    # Create dataset directory
    dataset_dir = Path(__file__).parent / "nasa"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dataset directory: {dataset_dir}")
    
    # NASA battery files
    # These are publicly available from NASA's data repository
    batteries = {
        "B0005": "Operated at 24°C (room temperature)",
        "B0006": "Operated at 24°C (room temperature)", 
        "B0007": "Operated at 24°C (room temperature)",
        "B0018": "Operated at 24°C (room temperature)",
    }
    
    # Note: NASA dataset is typically accessed through their portal
    # Direct download URLs may change. Check: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    
    base_url = "https://ti.arc.nasa.gov/c/6/"
    
    print("\n📊 Available batteries:")
    for battery, description in batteries.items():
        print(f"   {battery}: {description}")
    
    print("\n⚠️  IMPORTANT NOTICE:")
    print("   NASA dataset requires manual download from:")
    print("   https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print()
    print("   1. Visit the URL above")
    print("   2. Download 'Battery Data Set' (2.2 GB)")
    print("   3. Extract .mat files to:", dataset_dir)
    print()
    print("   Alternative: Use the Stanford/MIT dataset (easier to download)")
    
    # Check if files already exist
    existing_files = list(dataset_dir.glob("*.mat"))
    if existing_files:
        print(f"\n✅ Found {len(existing_files)} existing .mat files:")
        for f in existing_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n❌ No .mat files found. Please download manually.")
    
    return dataset_dir


def verify_dataset():
    """Verify downloaded dataset integrity."""
    dataset_dir = Path(__file__).parent / "nasa"
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        return False
    
    mat_files = list(dataset_dir.glob("*.mat"))
    
    if len(mat_files) == 0:
        print("❌ No .mat files found!")
        return False
    
    print(f"\n✅ Dataset verification passed!")
    print(f"   Found {len(mat_files)} battery files")
    
    total_size = sum(f.stat().st_size for f in mat_files)
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    
    return True


def convert_mat_to_csv():
    """Convert MATLAB files to CSV for easier processing."""
    try:
        from scipy.io import loadmat
        import pandas as pd
    except ImportError:
        print("⚠️ scipy not installed. Run: pip install scipy pandas")
        return
    
    dataset_dir = Path(__file__).parent / "nasa"
    mat_files = list(dataset_dir.glob("*.mat"))
    
    if not mat_files:
        print("❌ No .mat files to convert!")
        return
    
    csv_dir = dataset_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    
    print("\n🔄 Converting .mat files to CSV...")
    
    for mat_file in mat_files:
        print(f"   Processing {mat_file.name}...")
        
        try:
            # Load MATLAB file
            mat_data = loadmat(mat_file)
            battery_name = mat_file.stem
            
            # Extract cycle data
            cycles_data = []
            
            battery_key = battery_name  # Usually matches filename
            if battery_key not in mat_data:
                print(f"   ⚠️ Key '{battery_key}' not found in {mat_file.name}")
                continue
            
            cycles = mat_data[battery_key]['cycle'][0][0][0]
            
            for i, cycle in enumerate(cycles):
                try:
                    cycle_info = {
                        'battery': battery_name,
                        'cycle': i,
                        'type': cycle['type'][0],
                        'ambient_temp': cycle['ambient_temperature'][0][0],
                        'time': cycle['time'][0][0].flatten().tolist(),
                    }
                    
                    # Data may be in different structure
                    if 'data' in cycle.dtype.names:
                        data = cycle['data'][0][0]
                        cycle_info['voltage'] = data['Voltage_measured'][0][0].flatten().tolist()
                        cycle_info['current'] = data['Current_measured'][0][0].flatten().tolist()
                        cycle_info['temperature'] = data['Temperature_measured'][0][0].flatten().tolist()
                        
                        if 'Capacity' in data.dtype.names:
                            cycle_info['capacity'] = data['Capacity'][0][0][0][0]
                    
                    cycles_data.append(cycle_info)
                    
                except Exception as e:
                    print(f"      ⚠️ Error processing cycle {i}: {e}")
                    continue
            
            # Save to CSV (summary)
            summary_df = pd.DataFrame([{
                'battery': c['battery'],
                'cycle': c['cycle'],
                'type': c['type'],
                'ambient_temp': c['ambient_temp'],
                'capacity': c.get('capacity', None),
                'samples': len(c.get('time', []))
            } for c in cycles_data])
            
            csv_path = csv_dir / f"{battery_name}_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            print(f"   ✅ Saved {csv_path.name}")
            
        except Exception as e:
            print(f"   ❌ Error converting {mat_file.name}: {e}")
    
    print(f"\n✅ CSV files saved to: {csv_dir}")


if __name__ == "__main__":
    # Download dataset
    dataset_dir = download_nasa_battery_dataset()
    
    # Verify
    if verify_dataset():
        # Convert to CSV
        print("\n" + "="*60)
        convert_option = input("Convert .mat files to CSV? (y/n): ").strip().lower()
        if convert_option == 'y':
            convert_mat_to_csv()
    
    print("\n✅ Done!")
    print("\n📚 Next steps:")
    print("   1. Load data: from data_loader import BatteryDataLoader")
    print("   2. Extract features: loader.extract_features()")
    print("   3. Train model: python ml_models/train_degradation_model.py")
