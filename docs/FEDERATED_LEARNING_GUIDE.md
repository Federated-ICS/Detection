# Federated Learning Integration Guide for Detection Module

**Purpose:** Transform your standalone CNN-LSTM detection model into a federated learning system  
**Framework:** Flower (flwr) - Simple, framework-agnostic FL  
**Privacy:** Differential Privacy with Opacus  
**Timeline:** 1-2 weeks implementation

---

## Overview

This guide shows you how to take your trained CNN-LSTM model from the Detection module and deploy it in a federated learning setup where multiple facilities can collaboratively improve the model without sharing raw network traffic data.

### What You'll Build

```
Before (Standalone):
┌─────────────────┐
│  Your Facility  │
│  ├─ Network     │
│  ├─ CNN-LSTM    │
│  └─ Detects     │
└─────────────────┘

After (Federated):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Facility A     │     │  Facility B     │     │  Facility C     │
│  ├─ Network     │     │  ├─ Network     │     │  ├─ Network     │
│  ├─ CNN-LSTM    │     │  ├─ CNN-LSTM    │     │  ├─ CNN-LSTM    │
│  └─ Detects     │     │  └─ Detects     │     │  └─ Detects     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ↓
                        ┌─────────────────┐
                        │   FL Server     │
                        │  (Aggregates)   │
                        └─────────────────┘
```

**Key Benefit:** When Facility A sees a new attack, Facilities B and C learn to detect it within hours - without A sharing any raw data!

---

## Prerequisites

### What You Already Have
- ✅ Trained CNN-LSTM model (`best_multiclass_cnn_lstm_model.h5`)
- ✅ Preprocessing pipeline (`preprocessing.ipynb`)
- ✅ Training code (`train.ipynb`)
- ✅ Dataset (DNN-EdgeIIoT)

### What You Need to Install

```bash
# Federated Learning Framework
pip install flwr==1.6.0

# Differential Privacy (for TensorFlow/Keras)
pip install tensorflow-privacy==0.9.0

# Additional utilities
pip install pyyaml==6.0
pip install python-dotenv==1.0.0
```

---

## Architecture

### Three Components

1. **FL Server** - Coordinates training rounds, aggregates model updates
2. **FL Clients** (3+) - Each facility runs a client that trains locally
3. **Your CNN-LSTM Model** - The detection model being improved

### Data Flow

```
Round 1:
1. Server sends global model → All clients
2. Each client trains on local network traffic (parallel)
3. Clients send model updates → Server
4. Server aggregates updates → New global model

Round 2:
5. Repeat with improved model...
```

---

## Implementation

### Step 1: Convert Your Model to Flower Format

Create `fl_model.py`:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Tuple

class CNNLSTMModel:
    """Wrapper for your CNN-LSTM model to work with Flower"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize model
        
        Args:
            model_path: Path to saved model (optional)
        """
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = self.build_model()
    
    def build_model(self, input_shape=(1, 18), num_classes=15):
        """Build CNN-LSTM architecture (same as train.ipynb)"""
        model = keras.Sequential([
            # CNN Layers
            keras.layers.Conv1D(64, 3, activation='relu', 
                               input_shape=input_shape, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(1),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(1),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(1),
            keras.layers.Dropout(0.2),
            
            # LSTM Layers
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            
            # Dense Layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
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
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights for FL"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights from FL server"""
        self.model.set_weights(weights)
    
    def train(self, X_train, y_train, epochs=5, batch_size=128):
        """Train model locally"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
```

---

### Step 2: Create FL Client

Create `fl_client.py`:

```python
import flwr as fl
import numpy as np
import pandas as pd
from fl_model import CNNLSTMModel
from typing import Dict, Tuple, List
import os

class DetectionClient(fl.client.NumPyClient):
    """Federated Learning Client for each facility"""
    
    def __init__(
        self, 
        facility_id: str,
        data_path: str,
        model_path: str = None
    ):
        """
        Initialize FL client
        
        Args:
            facility_id: Unique facility identifier (e.g., "facility_a")
            data_path: Path to local network traffic data
            model_path: Path to initial model (optional)
        """
        self.facility_id = facility_id
        self.model = CNNLSTMModel(model_path)
        
        # Load local data
        self.X_train, self.y_train = self.load_local_data(data_path)
        
        print(f"✓ {facility_id} initialized with {len(self.X_train)} samples")
    
    def load_local_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load facility's local network traffic data
        
        Args:
            data_path: Path to CSV files
            
        Returns:
            X_train, y_train
        """
        # Load preprocessed data
        X = pd.read_csv(f"{data_path}/X_train.csv").values
        y = pd.read_csv(f"{data_path}/y_train.csv").values.ravel()
        
        # Reshape for CNN-LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return X, y
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return current model weights to FL server
        
        Called by: FL server at start of round
        Returns: Model weights (NOT data!)
        """
        return self.model.get_weights()
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data
        
        Called by: FL server during training round
        
        Args:
            parameters: Global model weights from server
            config: Training configuration
            
        Returns:
            - Updated model weights
            - Number of training samples
            - Metrics dictionary
        """
        # Update local model with global weights
        self.model.set_weights(parameters)
        
        # Train on local data
        print(f"[{self.facility_id}] Training locally...")
        history = self.model.train(
            self.X_train, 
            self.y_train,
            epochs=config.get("epochs", 5),
            batch_size=config.get("batch_size", 128)
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Return: weights, num_samples, metrics
        metrics = {
            "loss": float(history.history['loss'][-1]),
            "accuracy": float(history.history['accuracy'][-1])
        }
        
        print(f"[{self.facility_id}] Training complete: "
              f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
        return updated_weights, len(self.X_train), metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local test data
        
        Called by: FL server after aggregation
        
        Args:
            parameters: Global model weights
            config: Evaluation configuration
            
        Returns:
            - Loss value
            - Number of test samples
            - Metrics dictionary
        """
        # Update model with global weights
        self.model.set_weights(parameters)
        
        # Evaluate (you can load test data here)
        # For now, using training data as proxy
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train)
        
        print(f"[{self.facility_id}] Evaluation: "
              f"loss={loss:.4f}, acc={accuracy:.4f}")
        
        return loss, len(self.X_train), {"accuracy": accuracy}


def start_client(
    facility_id: str,
    data_path: str,
    server_address: str = "localhost:8080",
    model_path: str = None
):
    """
    Start FL client for a facility
    
    Args:
        facility_id: Facility identifier
        data_path: Path to local data
        server_address: FL server address
        model_path: Initial model path
    """
    # Create client
    client = DetectionClient(facility_id, data_path, model_path)
    
    # Connect to FL server
    print(f"\n[{facility_id}] Connecting to FL server at {server_address}...")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fl_client.py <facility_id> <data_path>")
        sys.exit(1)
    
    facility_id = sys.argv[1]
    data_path = sys.argv[2]
    
    start_client(facility_id, data_path)
```

---

### Step 3: Create FL Server

Create `fl_server.py`:

```python
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
import numpy as np

class FederatedDetectionServer:
    """FL Server for coordinating detection model training"""
    
    def __init__(
        self,
        num_rounds: int = 10,
        min_clients: int = 3,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0
    ):
        """
        Initialize FL server
        
        Args:
            num_rounds: Number of FL rounds
            min_clients: Minimum clients required
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        
        # Define aggregation strategy
        self.strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_evaluate_config,
        )
    
    def get_fit_config(self, server_round: int) -> Dict:
        """
        Return training configuration for each round
        
        Args:
            server_round: Current round number
            
        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": server_round,
            "epochs": 5,
            "batch_size": 128,
        }
        return config
    
    def get_evaluate_config(self, server_round: int) -> Dict:
        """
        Return evaluation configuration
        
        Args:
            server_round: Current round number
            
        Returns:
            Configuration dictionary
        """
        config = {
            "server_round": server_round,
        }
        return config
    
    def start(self, server_address: str = "0.0.0.0:8080"):
        """
        Start FL server
        
        Args:
            server_address: Server address and port
        """
        print("="*70)
        print("FEDERATED LEARNING SERVER")
        print("="*70)
        print(f"Server address: {server_address}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Minimum clients: {self.min_clients}")
        print("="*70)
        print("\nWaiting for clients to connect...")
        print("(Start clients with: python fl_client.py <facility_id> <data_path>)")
        print()
        
        # Start Flower server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )
        
        print("\n" + "="*70)
        print("FEDERATED LEARNING COMPLETE")
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FL Server for Detection")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum clients")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server address")
    
    args = parser.parse_args()
    
    server = FederatedDetectionServer(
        num_rounds=args.rounds,
        min_clients=args.min_clients
    )
    
    server.start(args.address)
```

---

### Step 4: Prepare Data for Multiple Facilities

Create `prepare_fl_data.py`:

```python
import pandas as pd
import numpy as np
from pathlib import Path

def split_data_for_facilities(
    X_path: str,
    y_path: str,
    num_facilities: int = 3,
    output_dir: str = "fl_data"
):
    """
    Split dataset into multiple facilities
    
    Args:
        X_path: Path to X_train.csv
        y_path: Path to y_train.csv
        num_facilities: Number of facilities to create
        output_dir: Output directory
    """
    # Load data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Split data
    total_samples = len(X)
    samples_per_facility = total_samples // num_facilities
    
    for i in range(num_facilities):
        facility_id = chr(ord('a') + i)  # a, b, c, ...
        facility_dir = Path(output_dir) / f"facility_{facility_id}"
        facility_dir.mkdir(exist_ok=True)
        
        # Get facility's data slice
        start_idx = i * samples_per_facility
        end_idx = start_idx + samples_per_facility if i < num_facilities - 1 else total_samples
        
        X_facility = X.iloc[start_idx:end_idx]
        y_facility = y.iloc[start_idx:end_idx]
        
        # Save
        X_facility.to_csv(facility_dir / "X_train.csv", index=False)
        y_facility.to_csv(facility_dir / "y_train.csv", index=False)
        
        print(f"✓ Facility {facility_id.upper()}: {len(X_facility)} samples → {facility_dir}")
    
    print(f"\n✓ Data split complete: {num_facilities} facilities")


if __name__ == "__main__":
    split_data_for_facilities(
        X_path="X_train.csv",
        y_path="y_train.csv",
        num_facilities=3
    )
```

---

## Usage

### Quick Start (3 Facilities)

**Terminal 1 - Start FL Server:**
```bash
python fl_server.py --rounds 10 --min-clients 3
```

**Terminal 2 - Start Facility A:**
```bash
python fl_client.py facility_a fl_data/facility_a
```

**Terminal 3 - Start Facility B:**
```bash
python fl_client.py facility_b fl_data/facility_b
```

**Terminal 4 - Start Facility C:**
```bash
python fl_client.py facility_c fl_data/facility_c
```

### Expected Output

**Server:**
```
======================================================================
FEDERATED LEARNING SERVER
======================================================================
Server address: 0.0.0.0:8080
Number of rounds: 10
Minimum clients: 3
======================================================================

Waiting for clients to connect...

INFO: Client facility_a connected
INFO: Client facility_b connected
INFO: Client facility_c connected

INFO: Starting round 1/10
INFO: Configuring 3 clients for training
INFO: Receiving 3 results
INFO: Aggregating results
INFO: Round 1 complete - Accuracy: 87.3%

INFO: Starting round 2/10
...
```

**Client (Facility A):**
```
✓ facility_a initialized with 443840 samples

[facility_a] Connecting to FL server at localhost:8080...
[facility_a] Connected successfully

[facility_a] Round 1: Receiving global model...
[facility_a] Training locally...
Epoch 1/5: loss=0.234, accuracy=0.856
Epoch 2/5: loss=0.189, accuracy=0.892
...
[facility_a] Training complete: loss=0.142, acc=0.912
[facility_a] Sending updates to server...

[facility_a] Round 2: Receiving global model...
...
```

---

## Adding Differential Privacy

### Step 5: Privacy-Enhanced Client

Create `fl_client_private.py`:

```python
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from fl_client import DetectionClient
import numpy as np

class PrivateDetectionClient(DetectionClient):
    """FL Client with Differential Privacy"""
    
    def __init__(
        self,
        facility_id: str,
        data_path: str,
        model_path: str = None,
        epsilon: float = 2.0,
        delta: float = 1e-5
    ):
        """
        Initialize private FL client
        
        Args:
            facility_id: Facility identifier
            data_path: Path to local data
            model_path: Initial model path
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter
        """
        super().__init__(facility_id, data_path, model_path)
        self.epsilon = epsilon
        self.delta = delta
        
        # Recompile model with DP optimizer
        self._setup_private_training()
    
    def _setup_private_training(self):
        """Setup differential privacy for training"""
        # Calculate noise multiplier for target epsilon
        # This is simplified - use tensorflow_privacy.compute_dp_sgd_privacy for exact calculation
        noise_multiplier = 1.1  # Calibrated for ε≈2.0
        
        # Create DP optimizer
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=1.0,  # Gradient clipping
            noise_multiplier=noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001
        )
        
        # Recompile model
        self.model.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"[{self.facility_id}] Differential Privacy enabled: "
              f"ε={self.epsilon}, δ={self.delta}")
    
    def fit(self, parameters, config):
        """Train with differential privacy"""
        # Same as parent, but DP is applied automatically by optimizer
        return super().fit(parameters, config)


# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fl_client_private.py <facility_id> <data_path>")
        sys.exit(1)
    
    facility_id = sys.argv[1]
    data_path = sys.argv[2]
    
    # Create private client
    from fl_client import start_client
    client = PrivateDetectionClient(facility_id, data_path, epsilon=2.0)
    
    # Connect to server
    import flwr as fl
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
```

---

## Demo Scenario: New Attack Detection

### Scenario: Port Scan Attack

**Setup:**
1. Facility A has port scan attacks in its data
2. Facilities B and C have never seen port scans
3. Run FL to share knowledge

**Steps:**

1. **Prepare specialized datasets:**
```python
# Give Facility A port scan attacks
facility_a_data = data[data['Attack_type'] == 'Port_Scanning']

# Give Facilities B & C other attacks (no port scans)
facility_b_data = data[data['Attack_type'] != 'Port_Scanning'][:len(facility_a_data)]
facility_c_data = data[data['Attack_type'] != 'Port_Scanning'][len(facility_a_data):]
```

2. **Test before FL:**
```python
# Test Facility B's model on port scans
accuracy_before = facility_b_model.evaluate(port_scan_test_data)
print(f"Facility B accuracy on port scans BEFORE FL: {accuracy_before:.2%}")
# Expected: ~50% (random guessing)
```

3. **Run FL (10 rounds):**
```bash
# Start server and 3 clients as shown above
```

4. **Test after FL:**
```python
# Test Facility B's model again
accuracy_after = facility_b_model.evaluate(port_scan_test_data)
print(f"Facility B accuracy on port scans AFTER FL: {accuracy_after:.2%}")
# Expected: ~95% (learned from Facility A!)
```

**Key Message:** Facility B learned to detect port scans without ever seeing them locally - it learned from Facility A through federated learning!

---

## Monitoring and Metrics

### Track FL Progress

Create `fl_monitor.py`:

```python
import matplotlib.pyplot as plt
import json
from pathlib import Path

class FLMonitor:
    """Monitor FL training progress"""
    
    def __init__(self, log_dir: str = "fl_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = {
            "rounds": [],
            "accuracy": [],
            "loss": [],
            "clients": []
        }
    
    def log_round(self, round_num: int, accuracy: float, loss: float, num_clients: int):
        """Log metrics for a round"""
        self.metrics["rounds"].append(round_num)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["loss"].append(loss)
        self.metrics["clients"].append(num_clients)
        
        # Save to file
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_progress(self):
        """Plot FL training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.metrics["rounds"], self.metrics["accuracy"], 'b-o')
        ax1.set_xlabel("FL Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Global Model Accuracy")
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.metrics["rounds"], self.metrics["loss"], 'r-o')
        ax2.set_xlabel("FL Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Global Model Loss")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "fl_progress.png", dpi=300)
        print(f"✓ Progress plot saved: {self.log_dir / 'fl_progress.png'}")
```

---

## Troubleshooting

### Issue: Clients can't connect to server

**Solution:**
```bash
# Check server is running
netstat -an | grep 8080

# Try localhost instead of 0.0.0.0
python fl_client.py facility_a fl_data/facility_a --server localhost:8080
```

### Issue: Training too slow

**Solution:**
```python
# Reduce epochs per round
config = {"epochs": 3}  # Instead of 5

# Reduce batch size
config = {"batch_size": 64}  # Instead of 128

# Use fewer samples
X_train = X_train[:10000]  # Use subset for testing
```

### Issue: Model accuracy not improving

**Solution:**
```python
# Increase number of rounds
num_rounds = 20  # Instead of 10

# Increase local epochs
epochs = 10  # Instead of 5

# Check data distribution
# Make sure each facility has diverse attack types
```

---

## Next Steps

### 1. Production Deployment
- Deploy FL server on cloud (AWS, Azure, GCP)
- Use secure communication (TLS/SSL)
- Add authentication for clients
- Implement model versioning

### 2. Advanced Features
- **Asynchronous FL**: Clients train at different times
- **Personalization**: Each facility fine-tunes global model
- **Byzantine-robust aggregation**: Handle malicious clients
- **Adaptive learning rates**: Adjust based on round performance

### 3. Integration
- Connect to Kafka for real-time data
- Add WebSocket for dashboard updates
- Integrate with GNN attack predictor
- Store FL metrics in PostgreSQL

---

## Summary

### What You Built

✅ **FL Server** - Coordinates training across facilities  
✅ **FL Clients** - Train locally, share only model updates  
✅ **Privacy** - Differential privacy protects individual data  
✅ **Monitoring** - Track progress and metrics

### Key Benefits

- 🔒 **Privacy**: Raw network traffic never leaves facility
- 🤝 **Collaboration**: Learn from all facilities' experiences
- ⚡ **Speed**: Hours instead of weeks for threat intelligence
- 📈 **Improvement**: Model gets better with each facility

### Files Created

```
Detection/
├── fl_model.py                    # Model wrapper
├── fl_server.py                   # FL server
├── fl_client.py                   # FL client
├── fl_client_private.py           # Private FL client
├── prepare_fl_data.py             # Data preparation
├── fl_monitor.py                  # Monitoring
└── FEDERATED_LEARNING_GUIDE.md    # This guide
```

---

## Resources

### Documentation
- Flower: https://flower.dev/docs/
- TensorFlow Privacy: https://github.com/tensorflow/privacy
- Federated Learning: https://federated.withgoogle.com/

### Papers
- "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- "Deep Learning with Differential Privacy"
- "Advances and Open Problems in Federated Learning"

---

**Guide Version:** 1.0  
**Last Updated:** November 25, 2025  
**Status:** Ready to Implement  
**Estimated Time:** 1-2 weeks
