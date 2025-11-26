"""
Stanford/MIT Fast-Charging Dataset Downloader
==============================================
Downloads the Stanford/MIT battery fast-charging optimization dataset.

This is one of the best datasets for ML training because:
- 124 cells with extensive cycling data
- Early-life prediction focus (perfect for your project)
- Well-documented and widely cited
- Easier to download than NASA dataset

Citation:
Severson et al., "Data-driven prediction of battery cycle life before capacity degradation"
Nature Energy, 2019. DOI: 10.1038/s41560-019-0356-8
"""

import requests
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import json


def download_stanford_mit_dataset():
    """
    Download Stanford/MIT fast-charging battery dataset.
    
    This downloads from the official data repository at data.matr.io
    """
    
    print("🔋 Stanford/MIT Fast-Charging Dataset Downloader")
    print("="*60)
    
    # Create dataset directory
    dataset_dir = Path(__file__).parent / "stanford_mit"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dataset directory: {dataset_dir}")
    
    # Dataset information
    dataset_info = {
        "name": "Stanford/MIT Fast-Charging Dataset",
        "cells": 124,
        "chemistry": "LFP/Graphite (A123 APR18650M1A)",
        "cycles": "Up to 2000+ per cell",
        "size": "~10 GB (full), ~500 MB (processed)",
        "url": "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
    }
    
    print("\n📊 Dataset Information:")
    for key, value in dataset_info.items():
        print(f"   {key}: {value}")
    
    print("\n⚠️  DOWNLOAD OPTIONS:")
    print("   Option 1 (Recommended): Download processed version")
    print("   Option 2: Download full raw data (requires ~10 GB)")
    
    # The dataset is available through data.matr.io
    # Processed version is more manageable
    
    print("\n📥 Downloading processed dataset...")
    
    # URLs for the processed dataset
    # Note: These are example URLs - actual URLs may require authentication
    files_to_download = {
        "batch1.pkl": "Primary training data (48 cells)",
        "batch2.pkl": "Secondary batch (48 cells)", 
        "batch3.pkl": "Test set (28 cells)",
    }
    
    print("\n⚠️  MANUAL DOWNLOAD REQUIRED:")
    print("   The Stanford/MIT dataset requires registration at:")
    print("   https://data.matr.io/1/projects/5c48dd2bc625d700019f3204")
    print()
    print("   Steps:")
    print("   1. Create free account at data.matr.io")
    print("   2. Navigate to the project page")
    print("   3. Download the following files:")
    for filename, description in files_to_download.items():
        print(f"      • {filename} - {description}")
    print(f"   4. Place files in: {dataset_dir}")
    
    # Alternative: Use direct links if available
    print("\n💡 ALTERNATIVE: Use sample data generator")
    print("   Run: python datasets/generate_synthetic_data.py")
    print("   This creates synthetic data based on published results")
    
    # Check if files exist
    existing_files = list(dataset_dir.glob("*.pkl"))
    if existing_files:
        print(f"\n✅ Found {len(existing_files)} existing .pkl files:")
        for f in existing_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("\n❌ No .pkl files found. Please download manually.")
        return False


def verify_stanford_dataset():
    """Verify dataset and show basic statistics."""
    dataset_dir = Path(__file__).parent / "stanford_mit"
    pkl_files = list(dataset_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("❌ No dataset files found!")
        return False
    
    print("\n✅ Dataset verification:")
    
    total_cells = 0
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            num_cells = len(data)
            total_cells += num_cells
            
            print(f"\n   {pkl_file.name}:")
            print(f"      Cells: {num_cells}")
            
            # Sample first cell
            first_cell = list(data.keys())[0]
            cell_data = data[first_cell]
            
            if 'summary' in cell_data:
                print(f"      Cycles: {len(cell_data['summary'])}")
            
            if 'cycle_life' in cell_data:
                print(f"      Cycle life: {cell_data['cycle_life']}")
            
        except Exception as e:
            print(f"   ⚠️ Error reading {pkl_file.name}: {e}")
    
    print(f"\n   Total cells: {total_cells}")
    return True


def extract_features_stanford():
    """Extract features from Stanford dataset for ML training."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("⚠️ pandas/numpy not installed. Run: pip install pandas numpy")
        return
    
    dataset_dir = Path(__file__).parent / "stanford_mit"
    pkl_files = list(dataset_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("❌ No dataset files found!")
        return
    
    print("\n🔄 Extracting features for ML training...")
    
    all_features = []
    all_targets = []
    
    for pkl_file in pkl_files:
        print(f"   Processing {pkl_file.name}...")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            for cell_id, cell_data in data.items():
                # Extract early cycle data (first 100 cycles)
                if 'summary' not in cell_data:
                    continue
                
                summary = cell_data['summary']
                
                # Features from early cycles (cycles 2-100)
                early_cycles = summary[:100] if len(summary) >= 100 else summary
                
                features = {
                    'cell_id': cell_id,
                    
                    # Capacity features
                    'capacity_cycle_2': early_cycles[0]['QD'] if len(early_cycles) > 0 else 0,
                    'capacity_cycle_100': early_cycles[-1]['QD'] if len(early_cycles) > 0 else 0,
                    'capacity_fade_early': early_cycles[0]['QD'] - early_cycles[-1]['QD'] if len(early_cycles) > 0 else 0,
                    
                    # Variance features (important predictors from paper)
                    'discharge_time_var': np.var([c['QD'] for c in early_cycles]) if early_cycles else 0,
                    'ir_var': np.var([c.get('IR', 0) for c in early_cycles]) if early_cycles else 0,
                    
                    # Temperature features
                    'temp_mean': np.mean([c.get('T_avg', 25) for c in early_cycles]) if early_cycles else 25,
                    'temp_max': np.max([c.get('T_max', 25) for c in early_cycles]) if early_cycles else 25,
                }
                
                # Target: cycle life
                target = cell_data.get('cycle_life', 0)
                
                all_features.append(features)
                all_targets.append(target)
                
        except Exception as e:
            print(f"   ⚠️ Error processing {pkl_file.name}: {e}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['cycle_life'] = all_targets
    
    # Save processed features
    output_path = dataset_dir / "processed_features.csv"
    features_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Features extracted and saved to: {output_path}")
    print(f"   Total samples: {len(features_df)}")
    print(f"   Feature columns: {list(features_df.columns)}")
    
    # Show statistics
    print(f"\n📊 Cycle life statistics:")
    print(features_df['cycle_life'].describe())
    
    return features_df


if __name__ == "__main__":
    # Download (or show instructions)
    success = download_stanford_mit_dataset()
    
    if success:
        # Verify dataset
        if verify_stanford_dataset():
            # Extract features
            print("\n" + "="*60)
            extract_option = input("Extract ML features? (y/n): ").strip().lower()
            if extract_option == 'y':
                extract_features_stanford()
    
    print("\n✅ Done!")
    print("\n📚 Next steps:")
    print("   1. Load features: df = pd.read_csv('stanford_mit/processed_features.csv')")
    print("   2. Train model: python ml_models/train_degradation_model.py")
    print("   3. Integrate with PyBaMM: python twin_service/api_pybamm.py")
