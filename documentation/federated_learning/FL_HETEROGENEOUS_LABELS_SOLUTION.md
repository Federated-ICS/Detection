# Federated Learning with Heterogeneous Label Spaces
## Handling Different Attack Types Across Facilities

**Problem:** Different facilities detect different attack types  
**Challenge:** Output layer dimensions don't match  
**Document Created:** November 28, 2025

---

## 1. The Problem

### Scenario

```
Facility A (Chemical Plant):
- Detects: 13 attack types
- Output layer: Dense(13, activation='softmax')
- Classes: [Normal, DDoS_UDP, DDoS_TCP, Port_Scanning, ...]

Facility B (Water Treatment):
- Detects: 15 attack types  
- Output layer: Dense(15, activation='softmax')
- Classes: [Normal, DDoS_UDP, MITM, Ransomware, XSS, ...]

Facility C (Power Plant):
- Detects: 11 attack types
- Output layer: Dense(11, activation='softmax')
- Classes: [Normal, DDoS_ICMP, SQL_injection, ...]
```

### Why This Breaks Standard FL

**Problem 1: Model Architecture Mismatch**
```python
# Facility A model
output_layer_A = Dense(13)  # 64 * 13 = 832 parameters

# Facility B model  
output_layer_B = Dense(15)  # 64 * 15 = 960 parameters

# Cannot aggregate! Different shapes!
```

**Problem 2: Label Encoding Mismatch**
```python
# Facility A
label_encoder_A.classes_ = ['Backdoor', 'DDoS_TCP', 'Normal', ...]
# 'DDoS_TCP' → encoded as 1

# Facility B
label_encoder_B.classes_ = ['DDoS_TCP', 'MITM', 'Normal', ...]
# 'DDoS_TCP' → encoded as 0

# Same attack, different encoding!
```

**Problem 3: Knowledge Transfer Failure**
- Facility A learns about Port_Scanning
- Facility B has no Port_Scanning class
- How does B benefit from A's knowledge?

---

## 2. Solution Approaches

### Solution 1: Global Label Space (Recommended) ⭐

**Concept:** All facilities use the same output layer with ALL possible attack types

#### Implementation

**Step 1: Define Global Label Space**

```python
# global_config.py

GLOBAL_ATTACK_TYPES = [
    'Normal',
    'Backdoor',
    'DDoS_HTTP',
    'DDoS_ICMP', 
    'DDoS_TCP',
    'DDoS_UDP',
    'Fingerprinting',
    'MITM',
    'Password',
    'Port_Scanning',
    'Ransomware',
    'SQL_injection',
    'Uploading',
    'Vulnerability_scanner',
    'XSS'
]

NUM_GLOBAL_CLASSES = len(GLOBAL_ATTACK_TYPES)  # 15
```

**Step 2: Unified Model Architecture**

```python
# fl_model_unified.py

def build_unified_cnn_lstm(input_shape, num_global_classes=15):
    """
    Build CNN-LSTM with FIXED output layer for all facilities
    """
    model = models.Sequential([
        # CNN Layers (same for all)
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # LSTM Layers (same for all)
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense Layers (same for all)
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # GLOBAL OUTPUT LAYER - Same for ALL facilities
        layers.Dense(num_global_classes, activation='softmax', name='global_output')
    ])
    
    return model
```

**Step 3: Global Label Encoder**

```python
# fl_label_encoder.py

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

class GlobalLabelEncoder:
    """
    Label encoder that maps local labels to global label space
    """
    
    def __init__(self, global_classes):
        """
        Args:
            global_classes: List of all possible attack types across all facilities
        """
        self.global_classes = np.array(global_classes)
        self.num_classes = len(global_classes)
        
        # Create mapping: class_name -> global_index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
        
    def transform(self, local_labels):
        """
        Transform local labels to global indices
        
        Args:
            local_labels: Array of attack type names (strings)
            
        Returns:
            Array of global indices
        """
        global_indices = np.array([
            self.class_to_idx[label] for label in local_labels
        ])
        return global_indices
    
    def inverse_transform(self, global_indices):
        """
        Transform global indices back to attack type names
        """
        return self.global_classes[global_indices]
    
    def get_local_mask(self, local_classes):
        """
        Get mask for classes present at this facility
        
        Args:
            local_classes: List of attack types present at this facility
            
        Returns:
            Boolean mask of shape (num_global_classes,)
        """
        mask = np.zeros(self.num_classes, dtype=bool)
        for cls in local_classes:
            if cls in self.class_to_idx:
                mask[self.class_to_idx[cls]] = True
        return mask


# Usage example
global_encoder = GlobalLabelEncoder(GLOBAL_ATTACK_TYPES)

# Facility A has only 13 attack types
facility_a_labels = ['Normal', 'DDoS_TCP', 'Port_Scanning', ...]
facility_a_encoded = global_encoder.transform(facility_a_labels)
# Returns: [0, 4, 9, ...] (global indices)

# Facility B has 15 attack types (all of them)
facility_b_labels = ['Normal', 'MITM', 'Ransomware', ...]
facility_b_encoded = global_encoder.transform(facility_b_labels)
# Returns: [0, 7, 10, ...] (global indices)
```

**Step 4: Modified Training with Class Weights**

```python
# fl_client_unified.py

import flwr as fl
import numpy as np
from tensorflow import keras

class UnifiedFLClient(fl.client.NumPyClient):
    """
    FL client that handles heterogeneous local labels
    """
    
    def __init__(self, facility_id, data_path, global_encoder):
        self.facility_id = facility_id
        self.global_encoder = global_encoder
        
        # Load local data
        self.X_train, self.y_train_local = self.load_local_data(data_path)
        
        # Transform local labels to global indices
        self.y_train_global = self.global_encoder.transform(self.y_train_local)
        
        # Identify which classes are present locally
        self.local_classes = np.unique(self.y_train_local)
        self.local_mask = self.global_encoder.get_local_mask(self.local_classes)
        
        # Build model with GLOBAL output layer
        self.model = build_unified_cnn_lstm(
            input_shape=(1, self.X_train.shape[2]),
            num_global_classes=self.global_encoder.num_classes
        )
        
        # Compile with class weights to handle missing classes
        self.compile_with_class_weights()
        
        print(f"✓ {facility_id} initialized")
        print(f"  Local classes: {len(self.local_classes)}/{self.global_encoder.num_classes}")
        print(f"  Classes: {self.local_classes}")
    
    def compile_with_class_weights(self):
        """
        Compile model with class weights to handle imbalanced/missing classes
        """
        # Calculate class weights for local classes
        unique_classes, counts = np.unique(self.y_train_global, return_counts=True)
        total_samples = len(self.y_train_global)
        
        # Create class weights dictionary
        class_weights = {}
        for cls_idx in range(self.global_encoder.num_classes):
            if cls_idx in unique_classes:
                # Present locally: weight inversely proportional to frequency
                count = counts[unique_classes == cls_idx][0]
                class_weights[cls_idx] = total_samples / (len(unique_classes) * count)
            else:
                # Not present locally: zero weight (won't affect loss)
                class_weights[cls_idx] = 0.0
        
        self.class_weights = class_weights
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, parameters, config):
        """
        Train model on local data with global label space
        """
        # Update model with global weights
        self.model.set_weights(parameters)
        
        # Train on local data with class weights
        print(f"[{self.facility_id}] Training locally...")
        history = self.model.fit(
            self.X_train,
            self.y_train_global,  # Using global indices!
            epochs=config.get("epochs", 5),
            batch_size=config.get("batch_size", 128),
            class_weight=self.class_weights,  # Handle missing classes
            verbose=0
        )
        
        # Return updated weights
        metrics = {
            "loss": float(history.history['loss'][-1]),
            "accuracy": float(history.history['accuracy'][-1]),
            "num_local_classes": len(self.local_classes)
        }
        
        print(f"[{self.facility_id}] Training complete: "
              f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate global model on local test data
        """
        self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(
            self.X_train, 
            self.y_train_global,
            verbose=0
        )
        
        return loss, len(self.X_train), {"accuracy": accuracy}
```

**Step 5: Server-Side Aggregation (No Changes Needed!)**

```python
# fl_server_unified.py

import flwr as fl

# Standard FedAvg works because all models have same architecture!
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

#### Advantages of Global Label Space

✅ **Model Architecture Consistency:** All facilities have identical models  
✅ **Simple Aggregation:** Standard FedAvg works without modification  
✅ **Knowledge Transfer:** Facility B can learn about attacks it hasn't seen  
✅ **Easy to Implement:** Minimal code changes  
✅ **Scalable:** Easy to add new attack types

#### Disadvantages

❌ **Unused Output Neurons:** Facilities with fewer attacks have unused outputs  
❌ **Memory Overhead:** Slightly larger models  
❌ **Class Imbalance:** Need class weights to handle missing classes

---

### Solution 2: Zero-Shot Learning with Embeddings

**Concept:** Learn attack representations instead of direct classification

#### Implementation

```python
# fl_model_embedding.py

def build_embedding_cnn_lstm(input_shape, embedding_dim=128):
    """
    Build CNN-LSTM that outputs embeddings instead of class probabilities
    """
    model = models.Sequential([
        # CNN Layers
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # LSTM Layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        
        # Embedding Layer (shared across all facilities)
        layers.Dense(embedding_dim, activation=None, name='embedding')
    ])
    
    return model


class EmbeddingClassifier:
    """
    Separate classifier head for each facility
    """
    
    def __init__(self, embedding_model, num_local_classes):
        self.embedding_model = embedding_model
        
        # Local classification head (NOT shared in FL)
        self.classifier = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_local_classes, activation='softmax')
        ])
    
    def predict(self, X):
        embeddings = self.embedding_model.predict(X)
        return self.classifier.predict(embeddings)
```

**FL Process:**
1. Only share embedding model weights (not classifier)
2. Each facility keeps its own classifier head
3. Embeddings learn general attack patterns
4. Local classifiers map embeddings to local classes

#### Advantages

✅ **Flexible Label Spaces:** Each facility can have different classes  
✅ **Transfer Learning:** Embeddings capture general attack patterns  
✅ **Smaller Shared Model:** Only embedding layers shared

#### Disadvantages

❌ **More Complex:** Requires two-stage training  
❌ **Less Direct:** Indirect knowledge transfer through embeddings  
❌ **Tuning Required:** Need to find good embedding dimension

---

### Solution 3: Hierarchical Classification

**Concept:** Use attack categories instead of specific types

#### Implementation

```python
# Attack hierarchy
ATTACK_HIERARCHY = {
    'Normal': 'Normal',
    'DDoS_UDP': 'DDoS',
    'DDoS_TCP': 'DDoS',
    'DDoS_ICMP': 'DDoS',
    'DDoS_HTTP': 'DDoS',
    'Port_Scanning': 'Reconnaissance',
    'Fingerprinting': 'Reconnaissance',
    'Vulnerability_scanner': 'Reconnaissance',
    'SQL_injection': 'Injection',
    'XSS': 'Injection',
    'Password': 'Credential_Access',
    'MITM': 'Credential_Access',
    'Backdoor': 'Persistence',
    'Ransomware': 'Impact',
    'Uploading': 'Exfiltration',
}

GLOBAL_CATEGORIES = [
    'Normal',
    'DDoS',
    'Reconnaissance',
    'Injection',
    'Credential_Access',
    'Persistence',
    'Impact',
    'Exfiltration'
]  # 8 categories

# Two-stage model
def build_hierarchical_model(input_shape):
    # Stage 1: Category classification (shared in FL)
    category_model = build_cnn_lstm(input_shape, num_classes=8)
    
    # Stage 2: Specific attack classification (local, not shared)
    # Each facility has its own stage 2 model
    
    return category_model
```

#### Advantages

✅ **Reduced Complexity:** Fewer output classes  
✅ **Better Generalization:** Categories are more universal  
✅ **MITRE ATT&CK Alignment:** Maps to attack tactics

#### Disadvantages

❌ **Loss of Specificity:** Can't distinguish between similar attacks  
❌ **Two-Stage Inference:** Slower prediction  
❌ **Requires Domain Knowledge:** Need to define hierarchy

---

## 3. Recommended Solution: Global Label Space

### Why This is Best for Your Use Case

1. **Simplicity:** Easiest to implement and maintain
2. **Standard FL:** Works with existing Flower framework
3. **Full Knowledge Transfer:** Facilities learn about all attack types
4. **Proven Approach:** Used in production FL systems

### Implementation Steps

**Step 1: Create Global Configuration**

```python
# config/global_config.py

GLOBAL_ATTACK_TYPES = [
    'Normal',
    'Backdoor',
    'DDoS_HTTP',
    'DDoS_ICMP',
    'DDoS_TCP',
    'DDoS_UDP',
    'Fingerprinting',
    'MITM',
    'Password',
    'Port_Scanning',
    'Ransomware',
    'SQL_injection',
    'Uploading',
    'Vulnerability_scanner',
    'XSS'
]

NUM_GLOBAL_CLASSES = 15
```

**Step 2: Update Data Preparation**

```python
# prepare_fl_data_unified.py

from config.global_config import GLOBAL_ATTACK_TYPES
from fl_label_encoder import GlobalLabelEncoder

def prepare_facility_data(facility_id, data_path):
    """
    Prepare data for a facility with global label encoding
    """
    # Load local data
    X = pd.read_csv(f'{data_path}/X_train.csv')
    y_local = pd.read_csv(f'{data_path}/y_train.csv').values.ravel()
    
    # Create global encoder
    global_encoder = GlobalLabelEncoder(GLOBAL_ATTACK_TYPES)
    
    # Transform to global indices
    y_global = global_encoder.transform(y_local)
    
    # Save
    np.save(f'{data_path}/y_train_global.npy', y_global)
    
    # Save local class info
    local_classes = np.unique(y_local)
    with open(f'{data_path}/local_classes.txt', 'w') as f:
        f.write('\n'.join(local_classes))
    
    print(f"✓ {facility_id}: {len(local_classes)}/{NUM_GLOBAL_CLASSES} classes")
    print(f"  Classes: {local_classes}")
```

**Step 3: Run FL**

```bash
# Terminal 1 - Server
python fl_server_unified.py --rounds 10 --min-clients 3

# Terminal 2 - Facility A (13 attack types)
python fl_client_unified.py facility_a fl_data/facility_a

# Terminal 3 - Facility B (15 attack types)
python fl_client_unified.py facility_b fl_data/facility_b

# Terminal 4 - Facility C (11 attack types)
python fl_client_unified.py facility_c fl_data/facility_c
```

---

## 4. Handling Edge Cases

### Case 1: New Attack Type Discovered

**Problem:** Facility D discovers a new attack type "Zero_Day"

**Solution:**
```python
# 1. Update global config
GLOBAL_ATTACK_TYPES.append('Zero_Day')
NUM_GLOBAL_CLASSES = 16

# 2. Retrain all models with new output layer
# (Can use transfer learning to preserve learned features)

# 3. Existing facilities get zero weight for new class
class_weights[15] = 0.0  # Zero_Day not present
```

### Case 2: Facility Joins Mid-Training

**Problem:** Facility E joins at round 5

**Solution:**
```python
# 1. Load current global model
global_model = load_model('global_model_round_5.h5')

# 2. Initialize Facility E with global model
facility_e_model.set_weights(global_model.get_weights())

# 3. Join FL from round 6 onwards
# (Facility E benefits from previous rounds immediately)
```

### Case 3: Extremely Rare Attack

**Problem:** Facility A has only 10 samples of "MITM"

**Solution:**
```python
# Use data augmentation
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)

# Or use higher class weight
class_weights[MITM_idx] = 10.0  # Emphasize rare class
```

---

## 5. Evaluation Strategy

### Per-Facility Evaluation

```python
def evaluate_facility(model, facility_id, X_test, y_test_local, global_encoder):
    """
    Evaluate model on facility's local test set
    """
    # Transform local labels to global
    y_test_global = global_encoder.transform(y_test_local)
    
    # Predict
    y_pred_proba = model.predict(X_test)
    y_pred_global = np.argmax(y_pred_proba, axis=1)
    
    # Convert back to local labels for interpretation
    y_pred_local = global_encoder.inverse_transform(y_pred_global)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_local, y_pred_local)
    
    # Per-class metrics (only for classes present locally)
    local_classes = np.unique(y_test_local)
    for cls in local_classes:
        mask = y_test_local == cls
        cls_accuracy = accuracy_score(
            y_test_local[mask], 
            y_pred_local[mask]
        )
        print(f"  {cls}: {cls_accuracy:.2%}")
    
    return accuracy
```

### Cross-Facility Evaluation (Knowledge Transfer)

```python
def evaluate_knowledge_transfer(model, facility_a_test, facility_b_test, global_encoder):
    """
    Test if Facility B learned about attacks only present in Facility A
    """
    # Get attacks unique to Facility A
    attacks_only_in_a = set(facility_a_test['classes']) - set(facility_b_test['classes'])
    
    print(f"\nKnowledge Transfer Test:")
    print(f"Attacks only in Facility A: {attacks_only_in_a}")
    
    # Test Facility B's model on Facility A's unique attacks
    for attack in attacks_only_in_a:
        # Get Facility A samples of this attack
        mask = facility_a_test['y_local'] == attack
        X_test = facility_a_test['X'][mask]
        y_test = facility_a_test['y_global'][mask]
        
        # Predict with Facility B's model
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Facility B accuracy on {attack}: {accuracy:.2%}")
        
        if accuracy > 0.7:
            print(f"    ✓ Knowledge transferred successfully!")
        else:
            print(f"    ✗ Limited knowledge transfer")
```

---

## 6. Complete Example

```python
# complete_fl_example_unified.py

from config.global_config import GLOBAL_ATTACK_TYPES, NUM_GLOBAL_CLASSES
from fl_label_encoder import GlobalLabelEncoder
from fl_client_unified import UnifiedFLClient
import flwr as fl

# Initialize global encoder
global_encoder = GlobalLabelEncoder(GLOBAL_ATTACK_TYPES)

# Start server (Terminal 1)
def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

# Start client (Terminal 2, 3, 4)
def start_client(facility_id, data_path):
    client = UnifiedFLClient(facility_id, data_path, global_encoder)
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    import sys
    
    if sys.argv[1] == "server":
        start_server()
    else:
        facility_id = sys.argv[1]
        data_path = sys.argv[2]
        start_client(facility_id, data_path)
```

---

## 7. Summary

### Problem
Different facilities have different attack types → Different output layer dimensions → Cannot aggregate models

### Solution
Use **Global Label Space** approach:
1. Define all possible attack types globally (15 classes)
2. All facilities use same model architecture with 15-class output
3. Use class weights to handle missing classes locally
4. Standard FedAvg aggregation works perfectly

### Benefits
- ✅ Simple implementation
- ✅ Full knowledge transfer
- ✅ Standard FL framework
- ✅ Scalable to new facilities
- ✅ Proven in production

### Trade-offs
- Slightly larger models (unused outputs)
- Need careful class weight tuning
- All facilities must agree on global label space

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** Complete  
**Recommended Approach:** Global Label Space (Solution 1)
