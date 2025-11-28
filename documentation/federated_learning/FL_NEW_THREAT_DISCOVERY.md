# Handling New Threat Discovery in Federated Learning
## Dynamic Label Space Expansion

**Problem:** What happens when a facility discovers a NEW attack type?  
**Challenge:** Global label space needs to expand dynamically  
**Document Created:** November 28, 2025

---

## 1. The Problem Scenario

### Initial State (Day 1)
```
Global Label Space: 15 attack types
- Normal, DDoS_UDP, DDoS_TCP, Port_Scanning, ...

All facilities trained with:
- Output layer: Dense(15, activation='softmax')
- Model parameters: 393,423
```

### New Threat Discovered (Day 30)
```
Facility B discovers: "Zero_Day_Exploit"

Problem:
- Global model has 15 outputs
- New attack is the 16th class
- Cannot classify new threat!
- Need to expand model architecture
```

---

## 2. Solution Approaches

### Solution 1: Model Expansion with Transfer Learning ⭐ (Recommended)

**Concept:** Expand output layer and transfer learned features

#### Step-by-Step Process

**Phase 1: Detect New Threat**
```python
# new_threat_detector.py

class NewThreatDetector:
    """
    Detect when a facility encounters unknown attack patterns
    """
    
    def __init__(self, model, global_encoder, confidence_threshold=0.6):
        self.model = model
        self.global_encoder = global_encoder
        self.confidence_threshold = confidence_threshold
    
    def detect_unknown(self, X_sample):
        """
        Detect if sample is potentially a new threat
        
        Returns:
            is_unknown: bool
            confidence: float
            predicted_class: str
        """
        # Get prediction
        pred_proba = self.model.predict(X_sample.reshape(1, 1, -1))
        max_confidence = np.max(pred_proba)
        predicted_idx = np.argmax(pred_proba)
        predicted_class = self.global_encoder.inverse_transform([predicted_idx])[0]
        
        # Low confidence suggests unknown attack
        is_unknown = max_confidence < self.confidence_threshold
        
        if is_unknown:
            print(f"⚠️  Potential new threat detected!")
            print(f"   Max confidence: {max_confidence:.2%}")
            print(f"   Closest match: {predicted_class}")
        
        return is_unknown, max_confidence, predicted_class
```

**Phase 2: Report New Threat**
```python
# threat_reporting.py

class ThreatReportingSystem:
    """
    System for facilities to report new threats to FL coordinator
    """
    
    def __init__(self, coordinator_url):
        self.coordinator_url = coordinator_url
    
    def report_new_threat(self, facility_id, threat_name, samples, metadata):
        """
        Report new threat discovery to FL coordinator
        
        Args:
            facility_id: ID of reporting facility
            threat_name: Proposed name for new threat
            samples: Sample data of new threat
            metadata: Additional information (timestamps, protocols, etc.)
        """
        report = {
            'facility_id': facility_id,
            'threat_name': threat_name,
            'num_samples': len(samples),
            'discovery_date': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        # Send to coordinator
        response = requests.post(
            f"{self.coordinator_url}/report_threat",
            json=report
        )
        
        if response.status_code == 200:
            print(f"✓ New threat '{threat_name}' reported successfully")
            print(f"  Assigned global index: {response.json()['new_index']}")
        
        return response.json()
```


**Phase 3: Expand Global Label Space**
```python
# global_config_manager.py

class GlobalConfigManager:
    """
    Manages dynamic expansion of global label space
    """
    
    def __init__(self, config_path='config/global_config.json'):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load current global configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.global_classes = config['global_classes']
        self.num_classes = len(self.global_classes)
        self.version = config['version']
    
    def add_new_threat(self, threat_name, reported_by, metadata):
        """
        Add new threat to global label space
        
        Args:
            threat_name: Name of new threat
            reported_by: Facility ID that discovered it
            metadata: Additional information
            
        Returns:
            new_index: Global index assigned to new threat
        """
        # Check if already exists
        if threat_name in self.global_classes:
            print(f"⚠️  Threat '{threat_name}' already exists")
            return self.global_classes.index(threat_name)
        
        # Add to global classes
        self.global_classes.append(threat_name)
        self.num_classes += 1
        new_index = self.num_classes - 1
        
        # Update version
        self.version += 1
        
        # Save updated config
        config = {
            'global_classes': self.global_classes,
            'num_classes': self.num_classes,
            'version': self.version,
            'last_updated': datetime.now().isoformat(),
            'threat_history': {
                threat_name: {
                    'discovered_by': reported_by,
                    'discovery_date': datetime.now().isoformat(),
                    'global_index': new_index,
                    'metadata': metadata
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ New threat added to global label space")
        print(f"  Threat: {threat_name}")
        print(f"  Global index: {new_index}")
        print(f"  Total classes: {self.num_classes}")
        print(f"  Config version: {self.version}")
        
        return new_index
```

**Phase 4: Expand Model Architecture**
```python
# model_expansion.py

def expand_model_output_layer(old_model, new_num_classes):
    """
    Expand output layer to accommodate new threat class
    
    Args:
        old_model: Existing model with N classes
        new_num_classes: N + 1 (or more)
        
    Returns:
        new_model: Model with expanded output layer
    """
    print(f"\n{'='*70}")
    print("EXPANDING MODEL ARCHITECTURE")
    print(f"{'='*70}")
    
    # Get old model configuration
    old_num_classes = old_model.layers[-1].output_shape[-1]
    print(f"Old output classes: {old_num_classes}")
    print(f"New output classes: {new_num_classes}")
    
    # Create new model with same architecture except output layer
    new_model = keras.Sequential()
    
    # Copy all layers except the last one
    for layer in old_model.layers[:-1]:
        new_model.add(layer)
    
    # Add new output layer with expanded dimensions
    new_model.add(
        keras.layers.Dense(
            new_num_classes,
            activation='softmax',
            name='expanded_output'
        )
    )
    
    # Transfer weights from old model
    print("\nTransferring weights...")
    for i, layer in enumerate(new_model.layers[:-1]):
        old_weights = old_model.layers[i].get_weights()
        layer.set_weights(old_weights)
        print(f"  ✓ Layer {i}: {layer.name}")
    
    # Initialize new output layer weights
    # Strategy: Extend old weights with small random values
    old_output_weights = old_model.layers[-1].get_weights()
    old_kernel, old_bias = old_output_weights
    
    # Old kernel shape: (64, old_num_classes)
    # New kernel shape: (64, new_num_classes)
    input_dim = old_kernel.shape[0]
    
    # Initialize new weights
    new_kernel = np.zeros((input_dim, new_num_classes))
    new_bias = np.zeros(new_num_classes)
    
    # Copy old weights
    new_kernel[:, :old_num_classes] = old_kernel
    new_bias[:old_num_classes] = old_bias
    
    # Initialize new class weights with small random values
    new_kernel[:, old_num_classes:] = np.random.randn(
        input_dim, 
        new_num_classes - old_num_classes
    ) * 0.01
    new_bias[old_num_classes:] = 0.0
    
    # Set new output layer weights
    new_model.layers[-1].set_weights([new_kernel, new_bias])
    print(f"  ✓ Output layer expanded: {old_num_classes} → {new_num_classes}")
    
    # Compile new model
    new_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n✓ Model expansion complete!")
    print(f"{'='*70}\n")
    
    return new_model
```


**Phase 5: Distribute Updated Model**
```python
# model_distribution.py

class ModelDistributionSystem:
    """
    Distribute updated model to all facilities
    """
    
    def __init__(self, facilities):
        self.facilities = facilities
    
    def distribute_expanded_model(self, new_model, new_config):
        """
        Send expanded model to all facilities
        
        Args:
            new_model: Model with expanded output layer
            new_config: Updated global configuration
        """
        print(f"\n{'='*70}")
        print("DISTRIBUTING EXPANDED MODEL TO FACILITIES")
        print(f"{'='*70}")
        
        # Save model
        model_path = f"models/global_model_v{new_config['version']}.h5"
        new_model.save(model_path)
        
        # Notify all facilities
        for facility_id in self.facilities:
            print(f"\nNotifying {facility_id}...")
            
            notification = {
                'type': 'model_update',
                'version': new_config['version'],
                'model_path': model_path,
                'config': new_config,
                'changes': {
                    'new_classes': new_config['global_classes'][-1:],
                    'total_classes': new_config['num_classes']
                }
            }
            
            # Send notification (in practice, use message queue or API)
            self.send_notification(facility_id, notification)
            
            print(f"  ✓ {facility_id} notified")
        
        print(f"\n✓ All facilities notified!")
        print(f"{'='*70}\n")
    
    def send_notification(self, facility_id, notification):
        """Send notification to facility (placeholder)"""
        # In production: use RabbitMQ, Kafka, or REST API
        pass
```

**Phase 6: Facility Update Process**
```python
# facility_update_handler.py

class FacilityUpdateHandler:
    """
    Handle model updates at facility side
    """
    
    def __init__(self, facility_id):
        self.facility_id = facility_id
    
    def handle_model_update(self, notification):
        """
        Handle incoming model update notification
        
        Args:
            notification: Update notification from coordinator
        """
        print(f"\n{'='*70}")
        print(f"[{self.facility_id}] RECEIVING MODEL UPDATE")
        print(f"{'='*70}")
        
        new_version = notification['version']
        model_path = notification['model_path']
        new_config = notification['config']
        
        print(f"New version: {new_version}")
        print(f"New classes: {notification['changes']['new_classes']}")
        print(f"Total classes: {notification['changes']['total_classes']}")
        
        # Download new model
        print("\nDownloading new model...")
        new_model = keras.models.load_model(model_path)
        print("  ✓ Model downloaded")
        
        # Update local configuration
        print("\nUpdating local configuration...")
        self.update_local_config(new_config)
        print("  ✓ Configuration updated")
        
        # Update label encoder
        print("\nUpdating label encoder...")
        self.global_encoder = GlobalLabelEncoder(new_config['global_classes'])
        print("  ✓ Label encoder updated")
        
        # Replace old model
        print("\nReplacing model...")
        self.model = new_model
        print("  ✓ Model replaced")
        
        # Update class weights (new classes get zero weight initially)
        print("\nUpdating class weights...")
        self.update_class_weights(new_config['num_classes'])
        print("  ✓ Class weights updated")
        
        print(f"\n✓ Update complete! Ready for next FL round.")
        print(f"{'='*70}\n")
    
    def update_class_weights(self, new_num_classes):
        """Update class weights to include new classes"""
        old_num_classes = len(self.class_weights)
        
        # Add zero weights for new classes (not present locally yet)
        for i in range(old_num_classes, new_num_classes):
            self.class_weights[i] = 0.0
        
        print(f"  Class weights: {old_num_classes} → {new_num_classes}")
```

---

## 3. Complete Workflow Example

### Scenario: Facility B Discovers "Zero_Day_Exploit"

**Day 30, 10:00 AM - Detection**
```python
# At Facility B
detector = NewThreatDetector(model, global_encoder)

# Unusual traffic detected
X_suspicious = capture_network_traffic()
is_unknown, confidence, closest = detector.detect_unknown(X_suspicious)

if is_unknown:
    print(f"⚠️  Unknown threat detected!")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Closest match: {closest}")
    
    # Collect samples
    samples = collect_threat_samples(X_suspicious, num_samples=100)
    
    # Report to coordinator
    reporter = ThreatReportingSystem("http://fl-coordinator:5000")
    response = reporter.report_new_threat(
        facility_id="facility_b",
        threat_name="Zero_Day_Exploit",
        samples=samples,
        metadata={
            'protocol': 'TCP',
            'target_port': 8080,
            'signature': 'CVE-2024-XXXXX'
        }
    )
```

**Day 30, 10:30 AM - Coordinator Processing**
```python
# At FL Coordinator
config_manager = GlobalConfigManager()

# Add new threat to global label space
new_index = config_manager.add_new_threat(
    threat_name="Zero_Day_Exploit",
    reported_by="facility_b",
    metadata=response['metadata']
)

# Output:
# ✓ New threat added to global label space
#   Threat: Zero_Day_Exploit
#   Global index: 15
#   Total classes: 16
#   Config version: 2
```


**Day 30, 11:00 AM - Model Expansion**
```python
# Load current global model
old_model = keras.models.load_model('models/global_model_v1.h5')

# Expand to 16 classes
new_model = expand_model_output_layer(old_model, new_num_classes=16)

# Output:
# ======================================================================
# EXPANDING MODEL ARCHITECTURE
# ======================================================================
# Old output classes: 15
# New output classes: 16
# 
# Transferring weights...
#   ✓ Layer 0: conv1d_1
#   ✓ Layer 1: batch_normalization
#   ...
#   ✓ Output layer expanded: 15 → 16
# 
# ✓ Model expansion complete!
```

**Day 30, 11:30 AM - Distribution**
```python
# Distribute to all facilities
distributor = ModelDistributionSystem(['facility_a', 'facility_b', 'facility_c'])

new_config = config_manager.load_config()
distributor.distribute_expanded_model(new_model, new_config)

# Output:
# ======================================================================
# DISTRIBUTING EXPANDED MODEL TO FACILITIES
# ======================================================================
# 
# Notifying facility_a...
#   ✓ facility_a notified
# 
# Notifying facility_b...
#   ✓ facility_b notified
# 
# Notifying facility_c...
#   ✓ facility_c notified
# 
# ✓ All facilities notified!
```

**Day 30, 12:00 PM - Facilities Update**
```python
# At each facility
handler = FacilityUpdateHandler('facility_a')

# Receive notification
notification = receive_update_notification()
handler.handle_model_update(notification)

# Output:
# ======================================================================
# [facility_a] RECEIVING MODEL UPDATE
# ======================================================================
# New version: 2
# New classes: ['Zero_Day_Exploit']
# Total classes: 16
# 
# Downloading new model...
#   ✓ Model downloaded
# 
# Updating local configuration...
#   ✓ Configuration updated
# 
# Updating label encoder...
#   ✓ Label encoder updated
# 
# Replacing model...
#   ✓ Model replaced
# 
# Updating class weights...
#   Class weights: 15 → 16
#   ✓ Class weights updated
# 
# ✓ Update complete! Ready for next FL round.
```

**Day 30, 2:00 PM - Resume FL Training**
```python
# FL Round 11 (first round with new threat)
# All facilities now have 16-class model

# Facility B (has Zero_Day_Exploit samples)
class_weights = {
    0: 1.0,   # Normal
    1: 1.2,   # Backdoor
    ...
    15: 2.0   # Zero_Day_Exploit (higher weight, rare)
}

# Facility A & C (don't have Zero_Day_Exploit yet)
class_weights = {
    0: 1.0,   # Normal
    1: 1.2,   # Backdoor
    ...
    15: 0.0   # Zero_Day_Exploit (zero weight, not present)
}

# FL continues normally!
# Facility A & C will learn about Zero_Day_Exploit through FL
```

---

## 4. Alternative: Incremental Learning Without Retraining

### Solution 2: Output Layer Fine-Tuning Only

**Concept:** Only retrain the output layer, freeze feature extractors

```python
def incremental_learning_approach(old_model, new_num_classes, new_threat_data):
    """
    Add new class without full retraining
    
    Args:
        old_model: Existing model
        new_num_classes: Expanded number of classes
        new_threat_data: Samples of new threat
    """
    # Expand model
    new_model = expand_model_output_layer(old_model, new_num_classes)
    
    # Freeze all layers except output
    for layer in new_model.layers[:-1]:
        layer.trainable = False
    
    # Only output layer is trainable
    new_model.layers[-1].trainable = True
    
    # Recompile
    new_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune on new threat data (quick!)
    X_new, y_new = new_threat_data
    new_model.fit(X_new, y_new, epochs=5, batch_size=32)
    
    # Unfreeze all layers for FL
    for layer in new_model.layers:
        layer.trainable = True
    
    return new_model
```

**Advantages:**
- ✅ Very fast (only output layer trained)
- ✅ Preserves learned features
- ✅ Minimal computational cost

**Disadvantages:**
- ❌ May not adapt features optimally for new threat
- ❌ Requires some samples of new threat

---

## 5. Handling Multiple Simultaneous Discoveries

### Scenario: Two Facilities Discover Different Threats

**Problem:**
```
Day 30, 10:00 AM:
- Facility A discovers "Threat_X"
- Facility B discovers "Threat_Y"

Both report simultaneously!
```

**Solution: Batch Updates**
```python
class BatchUpdateManager:
    """
    Handle multiple threat discoveries in batch
    """
    
    def __init__(self):
        self.pending_threats = []
        self.update_interval = 3600  # 1 hour
    
    def queue_threat(self, threat_report):
        """Add threat to pending queue"""
        self.pending_threats.append(threat_report)
        print(f"✓ Threat queued: {threat_report['threat_name']}")
        print(f"  Pending threats: {len(self.pending_threats)}")
    
    def process_batch_update(self):
        """Process all pending threats in one update"""
        if not self.pending_threats:
            return
        
        print(f"\n{'='*70}")
        print(f"PROCESSING BATCH UPDATE")
        print(f"{'='*70}")
        print(f"Number of new threats: {len(self.pending_threats)}")
        
        # Add all threats to global config
        config_manager = GlobalConfigManager()
        new_indices = []
        
        for threat in self.pending_threats:
            idx = config_manager.add_new_threat(
                threat['threat_name'],
                threat['facility_id'],
                threat['metadata']
            )
            new_indices.append(idx)
        
        # Expand model once for all new threats
        old_model = keras.models.load_model('models/global_model.h5')
        new_num_classes = config_manager.num_classes
        new_model = expand_model_output_layer(old_model, new_num_classes)
        
        # Distribute
        distributor = ModelDistributionSystem(facilities)
        distributor.distribute_expanded_model(new_model, config_manager.load_config())
        
        # Clear queue
        self.pending_threats = []
        
        print(f"\n✓ Batch update complete!")
        print(f"{'='*70}\n")
```

---

## 6. Versioning and Rollback Strategy

### Model Version Management

```python
class ModelVersionManager:
    """
    Manage model versions and enable rollback
    """
    
    def __init__(self, storage_path='models/'):
        self.storage_path = storage_path
        self.versions = []
    
    def save_version(self, model, config, version_number):
        """Save model version with metadata"""
        version_dir = f"{self.storage_path}/v{version_number}"
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model.save(f"{version_dir}/model.h5")
        
        # Save config
        with open(f"{version_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metadata
        metadata = {
            'version': version_number,
            'timestamp': datetime.now().isoformat(),
            'num_classes': config['num_classes'],
            'classes': config['global_classes']
        }
        
        with open(f"{version_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.versions.append(version_number)
        print(f"✓ Version {version_number} saved")
    
    def rollback(self, target_version):
        """Rollback to previous version"""
        print(f"\n⚠️  Rolling back to version {target_version}")
        
        version_dir = f"{self.storage_path}/v{target_version}"
        
        # Load model
        model = keras.models.load_model(f"{version_dir}/model.h5")
        
        # Load config
        with open(f"{version_dir}/config.json", 'r') as f:
            config = json.load(f)
        
        print(f"✓ Rollback complete")
        print(f"  Classes: {config['num_classes']}")
        
        return model, config
```



---

## 7. Practical Considerations

### Frequency of Updates

**Conservative Approach (Recommended):**
- Batch updates weekly or monthly
- Collect multiple new threats before expanding
- Reduces disruption to FL training

**Aggressive Approach:**
- Immediate expansion upon discovery
- Faster response to emerging threats
- More frequent model updates

### Validation Before Adding

```python
class ThreatValidator:
    """
    Validate new threat before adding to global space
    """
    
    def validate_threat(self, threat_report):
        """
        Validate that reported threat is genuinely new
        """
        checks = {
            'sufficient_samples': len(threat_report['samples']) >= 50,
            'distinct_from_existing': self.check_distinctness(threat_report),
            'verified_by_expert': threat_report.get('expert_verified', False),
            'reproducible': self.check_reproducibility(threat_report)
        }
        
        is_valid = all(checks.values())
        
        if not is_valid:
            print(f"⚠️  Threat validation failed:")
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check}")
        
        return is_valid
```

### Communication Protocol

```python
# Coordinator API endpoints

@app.route('/report_threat', methods=['POST'])
def report_threat():
    """Receive new threat report from facility"""
    data = request.json
    
    # Validate
    validator = ThreatValidator()
    if not validator.validate_threat(data):
        return {'error': 'Validation failed'}, 400
    
    # Add to global space
    config_manager = GlobalConfigManager()
    new_index = config_manager.add_new_threat(
        data['threat_name'],
        data['facility_id'],
        data['metadata']
    )
    
    # Trigger model expansion
    trigger_model_expansion()
    
    return {'new_index': new_index, 'status': 'accepted'}, 200


@app.route('/get_latest_model', methods=['GET'])
def get_latest_model():
    """Facilities can pull latest model"""
    version = request.args.get('version')
    
    latest_version = get_current_version()
    
    if version < latest_version:
        model_path = f"models/global_model_v{latest_version}.h5"
        config_path = f"config/global_config_v{latest_version}.json"
        
        return {
            'update_available': True,
            'latest_version': latest_version,
            'model_url': f"/download/model/{latest_version}",
            'config_url': f"/download/config/{latest_version}"
        }
    
    return {'update_available': False}
```

---

## 8. Testing Strategy

### Simulated New Threat Discovery

```python
# test_new_threat.py

def test_new_threat_workflow():
    """
    Test complete workflow of new threat discovery
    """
    print("="*70)
    print("TESTING NEW THREAT DISCOVERY WORKFLOW")
    print("="*70)
    
    # Step 1: Initial state
    print("\n1. Initial State")
    config = GlobalConfigManager()
    print(f"   Current classes: {config.num_classes}")
    
    # Step 2: Simulate discovery
    print("\n2. Simulating New Threat Discovery")
    new_threat_data = generate_synthetic_threat_data()
    
    # Step 3: Report
    print("\n3. Reporting to Coordinator")
    reporter = ThreatReportingSystem("http://localhost:5000")
    response = reporter.report_new_threat(
        facility_id="test_facility",
        threat_name="Test_Threat",
        samples=new_threat_data,
        metadata={'test': True}
    )
    
    # Step 4: Verify expansion
    print("\n4. Verifying Model Expansion")
    config.load_config()
    assert config.num_classes == 16, "Model not expanded!"
    print(f"   ✓ Classes expanded: 15 → {config.num_classes}")
    
    # Step 5: Test prediction
    print("\n5. Testing Prediction on New Threat")
    new_model = keras.models.load_model('models/global_model_v2.h5')
    pred = new_model.predict(new_threat_data[:1])
    predicted_class = np.argmax(pred)
    print(f"   Predicted class: {predicted_class}")
    print(f"   Expected class: 15 (new threat)")
    
    print("\n" + "="*70)
    print("✓ TEST PASSED")
    print("="*70)
```

---

## 9. Summary

### The Problem
When a facility discovers a NEW attack type, the global model with fixed 15-class output cannot handle it.

### The Solution
**Dynamic Model Expansion:**

1. **Detect** new threat at facility (low confidence predictions)
2. **Report** to FL coordinator with samples
3. **Validate** that it's genuinely new
4. **Expand** global label space (15 → 16 classes)
5. **Expand** model output layer (transfer old weights)
6. **Distribute** updated model to all facilities
7. **Resume** FL training with expanded model

### Key Benefits
✅ **No retraining from scratch** - Transfer learned features  
✅ **Minimal downtime** - Quick expansion process  
✅ **Backward compatible** - Old facilities can update seamlessly  
✅ **Knowledge preserved** - All previous learning retained  
✅ **Scalable** - Can handle multiple new threats

### Best Practices
- Batch updates to reduce disruption
- Validate new threats before adding
- Version all models and configs
- Enable rollback capability
- Test expansion process regularly

### Trade-offs
- **Slight disruption** during model update
- **Coordination overhead** for distribution
- **Storage requirements** for multiple versions
- **Validation complexity** to avoid false positives

---

## 10. Complete Code Example

```python
# complete_dynamic_fl.py

# 1. Setup
config_manager = GlobalConfigManager()
model_expander = ModelExpansionSystem()
distributor = ModelDistributionSystem(['facility_a', 'facility_b', 'facility_c'])

# 2. Normal FL training (Rounds 1-10)
for round_num in range(1, 11):
    run_fl_round(round_num)

# 3. New threat discovered at Round 10
new_threat_report = {
    'facility_id': 'facility_b',
    'threat_name': 'Zero_Day_Exploit',
    'samples': collected_samples,
    'metadata': {'protocol': 'TCP', 'port': 8080}
}

# 4. Expand model
new_index = config_manager.add_new_threat(
    new_threat_report['threat_name'],
    new_threat_report['facility_id'],
    new_threat_report['metadata']
)

old_model = keras.models.load_model('models/global_model_v1.h5')
new_model = model_expander.expand_model_output_layer(old_model, 16)

# 5. Distribute
distributor.distribute_expanded_model(new_model, config_manager.load_config())

# 6. Resume FL training (Rounds 11-20)
for round_num in range(11, 21):
    run_fl_round(round_num)  # Now with 16 classes!

print("✓ FL training complete with dynamic threat discovery!")
```

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** Complete  
**Recommended Approach:** Model Expansion with Transfer Learning
