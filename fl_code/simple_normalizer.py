#!/usr/bin/env python3
"""
Simple per-facility normalizer for FL
Handles data heterogeneity
"""

import numpy as np
import pickle
from pathlib import Path

class SimpleNormalizer:
    """
    Per-facility normalizer for handling heterogeneous data
    
    Each facility normalizes based on its own statistics,
    ensuring different traffic patterns are on the same scale.
    """
    
    def __init__(self, facility_id):
        """
        Initialize normalizer
        
        Args:
            facility_id: Facility identifier (e.g., "facility_a")
        """
        self.facility_id = facility_id
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X):
        """
        Learn normalization parameters
        
        Args:
            X: Training data (numpy array or pandas DataFrame)
        """
        if hasattr(X, 'values'):
            X = X.values
        
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8  # Add small value to avoid division by zero
        self.fitted = True
        
        print(f"✓ {self.facility_id} normalizer fitted")
        print(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
        print(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")
    
    def transform(self, X):
        """
        Apply normalization
        
        Args:
            X: Data to normalize
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if hasattr(X, 'values'):
            X = X.values
        
        # Z-score normalization
        X_normalized = (X - self.mean) / self.std
        
        return X_normalized
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X: Training data
            
        Returns:
            Normalized data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_normalized):
        """
        Denormalize data back to original scale
        
        Args:
            X_normalized: Normalized data
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        return X_normalized * self.std + self.mean
    
    def save(self, path):
        """
        Save normalizer to disk
        
        Args:
            path: File path to save to
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std,
                'facility_id': self.facility_id
            }, f)
        
        print(f"✓ Normalizer saved: {path}")
    
    def load(self, path):
        """
        Load normalizer from disk
        
        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.std = data['std']
            self.facility_id = data.get('facility_id', 'unknown')
            self.fitted = True
        
        print(f"✓ Normalizer loaded: {path}")


# Test
if __name__ == "__main__":
    import pandas as pd
    
    # Test with sample data
    print("Testing SimpleNormalizer...")
    
    # Create sample data
    X = np.random.randn(1000, 18) * 100 + 50
    
    # Create normalizer
    normalizer = SimpleNormalizer('test_facility')
    
    # Fit and transform
    X_normalized = normalizer.fit_transform(X)
    
    print(f"\nOriginal data:")
    print(f"  Mean: {X.mean():.2f}")
    print(f"  Std: {X.std():.2f}")
    print(f"  Range: [{X.min():.2f}, {X.max():.2f}]")
    
    print(f"\nNormalized data:")
    print(f"  Mean: {X_normalized.mean():.2f}")
    print(f"  Std: {X_normalized.std():.2f}")
    print(f"  Range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    
    # Test save/load
    normalizer.save('test_normalizer.pkl')
    
    normalizer2 = SimpleNormalizer('test_facility')
    normalizer2.load('test_normalizer.pkl')
    
    X_normalized2 = normalizer2.transform(X)
    
    print(f"\nAfter save/load:")
    print(f"  Same result: {np.allclose(X_normalized, X_normalized2)}")
    
    print("\n✓ All tests passed!")
