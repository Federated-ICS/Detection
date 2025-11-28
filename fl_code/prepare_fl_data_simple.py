#!/usr/bin/env python3
"""
Simple data preparation for FL demo
Split dataset into 3 facilities
"""

import pandas as pd
import numpy as np
from pathlib import Path

def prepare_fl_data(
    X_path='X_train.csv',
    y_path='y_train.csv',
    num_facilities=3,
    output_dir='fl_data',
    sample_size=None  # Use None for full dataset, or 10000 for testing
):
    """
    Split dataset into multiple facilities
    
    Args:
        X_path: Path to features CSV
        y_path: Path to labels CSV
        num_facilities: Number of facilities (default: 3)
        output_dir: Output directory
        sample_size: Number of samples to use (None = all)
    """
    print("="*60)
    print("PREPARING FL DATA")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    
    # Sample if requested
    if sample_size and sample_size < len(X):
        print(f"\n2. Sampling {sample_size} samples for faster testing...")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Split data
    print(f"\n3. Splitting into {num_facilities} facilities...")
    samples_per_facility = len(X) // num_facilities
    
    for i in range(num_facilities):
        facility_id = chr(ord('a') + i)  # a, b, c, ...
        facility_dir = Path(output_dir) / f"facility_{facility_id}"
        facility_dir.mkdir(exist_ok=True)
        
        # Get facility's data slice
        start_idx = i * samples_per_facility
        if i < num_facilities - 1:
            end_idx = start_idx + samples_per_facility
        else:
            end_idx = len(X)  # Last facility gets remaining samples
        
        X_facility = X.iloc[start_idx:end_idx]
        y_facility = y.iloc[start_idx:end_idx]
        
        # Save
        X_facility.to_csv(facility_dir / 'X_train.csv', index=False)
        y_facility.to_csv(facility_dir / 'y_train.csv', index=False)
        
        # Count attacks
        num_attacks = (y_facility.values > 0).sum()
        attack_ratio = num_attacks / len(y_facility) * 100
        
        print(f"   ✓ Facility {facility_id.upper()}: {len(X_facility)} samples "
              f"({num_attacks} attacks, {attack_ratio:.1f}%)")
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Output directory: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare FL data')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples (None = all)')
    parser.add_argument('--facilities', type=int, default=3,
                       help='Number of facilities')
    
    args = parser.parse_args()
    
    prepare_fl_data(
        sample_size=args.samples,
        num_facilities=args.facilities
    )
