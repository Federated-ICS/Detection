#!/usr/bin/env python3
"""
Simple Federated Learning Example
==================================

This is a minimal working example of FL with your CNN-LSTM model.
Perfect for understanding the basics before diving into the full implementation.

Usage:
    # Terminal 1 - Server
    python fl_simple_example.py server
    
    # Terminal 2 - Client A
    python fl_simple_example.py client facility_a
    
    # Terminal 3 - Client B
    python fl_simple_example.py client facility_b
    
    # Terminal 4 - Client C
    python fl_simple_example.py client facility_c
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from typing import List, Tuple, Dict


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_cnn_lstm_model(input_shape=(1, 18), num_classes=15):
    """Create CNN-LSTM model (simplified version)"""
    model = keras.Sequential([
        # CNN Layers
        keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # LSTM Layers
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        
        # Dense Layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # Output
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def load_facility_data(facility_id: str):
    """
    Load data for a specific facility
    
    Args:
        facility_id: 'facility_a', 'facility_b', or 'facility_c'
    
    Returns:
        X_train, y_train (numpy arrays)
    """
    try:
        # Try to load from fl_data directory
        X = pd.read_csv(f'fl_data/{facility_id}/X_train.csv').values
        y = pd.read_csv(f'fl_data/{facility_id}/y_train.csv').values.ravel()
    except FileNotFoundError:
        # Fallback: Load full dataset and split
        print(f"⚠️  fl_data/{facility_id} not found, using subset of main dataset")
        X = pd.read_csv('X_train.csv').values
        y = pd.read_csv('y_train.csv').values.ravel()
        
        # Split into 3 parts
        n = len(X) // 3
        if facility_id == 'facility_a':
            X, y = X[:n], y[:n]
        elif facility_id == 'facility_b':
            X, y = X[n:2*n], y[n:2*n]
        else:  # facility_c
            X, y = X[2*n:], y[2*n:]
    
    # Reshape for CNN-LSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    print(f"✓ Loaded {len(X)} samples for {facility_id}")
    return X, y


# ============================================================================
# FEDERATED LEARNING CLIENT
# ============================================================================

class SimpleFlowerClient(fl.client.NumPyClient):
    """Simple FL client for demonstration"""
    
    def __init__(self, facility_id: str):
        self.facility_id = facility_id
        self.model = create_cnn_lstm_model()
        self.X_train, self.y_train = load_facility_data(facility_id)
        
        print(f"\n{'='*60}")
        print(f"FL CLIENT: {facility_id.upper()}")
        print(f"{'='*60}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Model parameters: {self.model.count_params():,}")
        print(f"{'='*60}\n")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model weights"""
        return self.model.get_weights()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        # Update model with global weights
        self.model.set_weights(parameters)
        
        # Get training config
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 128)
        
        # Train
        print(f"\n[{self.facility_id}] 🚀 Training locally...")
        print(f"[{self.facility_id}] Epochs: {epochs}, Batch size: {batch_size}")
        
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # Silent training
        )
        
        # Get final metrics
        loss = float(history.history['loss'][-1])
        accuracy = float(history.history['accuracy'][-1])
        
        print(f"[{self.facility_id}] ✓ Training complete!")
        print(f"[{self.facility_id}]   Loss: {loss:.4f}")
        print(f"[{self.facility_id}]   Accuracy: {accuracy:.4f}")
        
        # Return updated weights, number of samples, and metrics
        return self.model.get_weights(), len(self.X_train), {
            "loss": loss,
            "accuracy": accuracy
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate global model"""
        # Update model with global weights
        self.model.set_weights(parameters)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        
        print(f"[{self.facility_id}] 📊 Evaluation:")
        print(f"[{self.facility_id}]   Loss: {loss:.4f}")
        print(f"[{self.facility_id}]   Accuracy: {accuracy:.4f}")
        
        return loss, len(self.X_train), {"accuracy": accuracy}


# ============================================================================
# FEDERATED LEARNING SERVER
# ============================================================================

def start_server(num_rounds: int = 5, min_clients: int = 3):
    """Start FL server"""
    print("\n" + "="*70)
    print("FEDERATED LEARNING SERVER")
    print("="*70)
    print(f"Number of rounds: {num_rounds}")
    print(f"Minimum clients: {min_clients}")
    print("="*70)
    print("\n⏳ Waiting for clients to connect...")
    print("   (Start clients in separate terminals)")
    print()
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE! 🎉")
    print("="*70)


def start_client(facility_id: str, server_address: str = "localhost:8080"):
    """Start FL client"""
    # Create client
    client = SimpleFlowerClient(facility_id)
    
    # Connect to server
    print(f"🔌 Connecting to FL server at {server_address}...")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python fl_simple_example.py server [rounds] [min_clients]")
        print("  Client: python fl_simple_example.py client <facility_id> [server_address]")
        print()
        print("Examples:")
        print("  python fl_simple_example.py server 5 3")
        print("  python fl_simple_example.py client facility_a")
        print("  python fl_simple_example.py client facility_b localhost:8080")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        # Server mode
        num_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        min_clients = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        start_server(num_rounds, min_clients)
    
    elif mode == "client":
        # Client mode
        if len(sys.argv) < 3:
            print("Error: Please specify facility_id")
            print("Usage: python fl_simple_example.py client <facility_id>")
            sys.exit(1)
        
        facility_id = sys.argv[2]
        server_address = sys.argv[3] if len(sys.argv) > 3 else "localhost:8080"
        start_client(facility_id, server_address)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Use 'server' or 'client'")
        sys.exit(1)


if __name__ == "__main__":
    main()
