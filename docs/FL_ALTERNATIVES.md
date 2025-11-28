# Alternatives to FedProx for Handling Heterogeneity

**Created:** November 25, 2025  
**Purpose:** Compare different approaches to handle data heterogeneity in FL

---

## 📊 Quick Comparison Table

| Method | Complexity | Effectiveness | Implementation Time | Best For |
|--------|------------|---------------|---------------------|----------|
| **FedProx** | ⭐⭐ Medium | ⭐⭐⭐⭐ High | 2-3 hours | General heterogeneity |
| **FedAvg** | ⭐ Easy | ⭐⭐ Low | 1 hour | Homogeneous data |
| **FedBN** | ⭐ Easy | ⭐⭐⭐ Medium | 1-2 hours | Different distributions |
| **SCAFFOLD** | ⭐⭐⭐ Hard | ⭐⭐⭐⭐⭐ Very High | 1 day | High heterogeneity |
| **FedNova** | ⭐⭐ Medium | ⭐⭐⭐⭐ High | 3-4 hours | Different training speeds |
| **Per-FedAvg** | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ High | 1 day | Personalization needed |
| **Clustered FL** | ⭐⭐ Medium | ⭐⭐⭐ Medium | 4-6 hours | Similar facility groups |

---

## 1. FedAvg (Baseline)

### **What It Is**
Standard federated averaging - simple weighted average of client models.

### **How It Works**
```python
# Server aggregates client updates
global_weights = sum(n_i * w_i for all clients) / sum(n_i)

# Where:
# n_i = number of samples at client i
# w_i = model weights from client i
```

### **Pros**
- ✅ Simple to implement
- ✅ Fast
- ✅ Works well with homogeneous data

### **Cons**
- ❌ Poor performance with heterogeneous data
- ❌ Models can diverge
- ❌ Unfair to minority clients

### **When to Use**
- All facilities have similar data
- Quick testing/prototyping
- Baseline comparison

### **Code Example**
```python
def fedavg_aggregate(client_weights, client_sizes):
    """Standard FedAvg aggregation"""
    total_size = sum(client_sizes)
    
    # Weighted average
    global_weights = []
    for layer_idx in range(len(client_weights[0])):
        layer = sum(
            (size / total_size) * weights[layer_idx]
            for weights, size in zip(client_weights, client_sizes)
        )
        global_weights.append(layer)
    
    return global_weights
```

---

## 2. FedBN (Federated Batch Normalization)

### **What It Is**
Keep BatchNorm layers local, only share other layers.

### **How It Works**
```python
# Each facility keeps its own BatchNorm statistics
# Only share Conv/LSTM/Dense layers

Facility A:
- BatchNorm stats: mean=100, std=20 (local)
- Conv/LSTM weights: shared with others

Facility B:
- BatchNorm stats: mean=50, std=10 (local)
- Conv/LSTM weights: shared with others
```

### **Pros**
- ✅ Very simple to implement
- ✅ Handles different data distributions well
- ✅ No hyperparameters to tune
- ✅ Works with existing models

### **Cons**
- ❌ Only helps with distribution shift
- ❌ Doesn't handle label heterogeneity
- ❌ Requires BatchNorm layers

### **When to Use**
- Different network traffic patterns (your case!)
- Model has BatchNorm layers
- Quick improvement over FedAvg

### **Code Example**
```python
class FedBNClient:
    """FL client with local BatchNorm"""
    
    def get_parameters(self):
        """Return only non-BatchNorm parameters"""
        params = []
        for layer in self.model.layers:
            if not isinstance(layer, BatchNormalization):
                params.append(layer.get_weights())
        return params
    
    def set_parameters(self, params):
        """Set only non-BatchNorm parameters"""
        param_idx = 0
        for layer in self.model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.set_weights(params[param_idx])
                param_idx += 1
            # BatchNorm layers keep local statistics
```

### **Expected Improvement**
```
FedAvg:  68% average accuracy
FedBN:   78% average accuracy (+10%)
FedProx: 82% average accuracy (+14%)
```

---

## 3. SCAFFOLD (Stochastic Controlled Averaging)

### **What It Is**
Use control variates to correct for client drift.

### **How It Works**
```python
# Track "drift" of each client
# Correct for drift during aggregation

Client drift = How much client's gradient differs from global

Server maintains:
- Global model
- Global control variate (average drift)

Each client maintains:
- Local model
- Local control variate (local drift)

Update rule:
local_update = gradient - local_drift + global_drift
```

### **Pros**
- ✅ Best performance with high heterogeneity
- ✅ Faster convergence than FedProx
- ✅ Theoretically sound

### **Cons**
- ❌ Complex to implement
- ❌ Requires storing control variates
- ❌ More communication overhead
- ❌ Harder to debug

### **When to Use**
- Very high heterogeneity
- Need best possible performance
- Have time for complex implementation

### **Code Example**
```python
class SCAFFOLDClient:
    """SCAFFOLD client with control variates"""
    
    def __init__(self):
        self.model = create_model()
        self.c_local = [np.zeros_like(w) for w in self.model.get_weights()]
        self.c_global = None
    
    def train(self, global_weights, c_global):
        """Train with drift correction"""
        self.c_global = c_global
        
        # Standard training
        for batch in data:
            # Compute gradient
            grad = compute_gradient(batch)
            
            # Correct for drift
            corrected_grad = grad - self.c_local + self.c_global
            
            # Update model
            self.model.apply_gradients(corrected_grad)
        
        # Update local control variate
        self.c_local = self.compute_drift()
        
        return self.model.get_weights(), self.c_local
```

### **Expected Improvement**
```
FedAvg:    68% average accuracy
FedProx:   82% average accuracy
SCAFFOLD:  88% average accuracy (+20% over FedAvg)
```

---

## 4. FedNova (Federated Normalized Averaging)

### **What It Is**
Normalize client updates by number of local steps.

### **How It Works**
```python
# Problem: Clients train for different numbers of steps
Facility A: 100 steps (fast GPU)
Facility B: 50 steps (slow CPU)
Facility C: 200 steps (very fast GPU)

# FedAvg treats them equally → bias toward C

# FedNova normalizes by steps:
normalized_update_A = update_A / 100
normalized_update_B = update_B / 50
normalized_update_C = update_C / 200

# Now fair aggregation
```

### **Pros**
- ✅ Handles different training speeds
- ✅ Fair to all clients
- ✅ Simple to implement

### **Cons**
- ❌ Only addresses step heterogeneity
- ❌ Doesn't help with data heterogeneity
- ❌ Requires tracking local steps

### **When to Use**
- Clients have different hardware
- Different training speeds
- Combine with FedProx for best results

### **Code Example**
```python
class FedNovaClient:
    """FedNova client with step normalization"""
    
    def train(self, global_weights, epochs=5):
        """Train and track steps"""
        self.model.set_weights(global_weights)
        
        total_steps = 0
        for epoch in range(epochs):
            for batch in data:
                self.model.train_on_batch(batch)
                total_steps += 1
        
        # Compute normalized update
        update = self.model.get_weights() - global_weights
        normalized_update = [u / total_steps for u in update]
        
        return normalized_update, total_steps

class FedNovaServer:
    """FedNova server with normalized aggregation"""
    
    def aggregate(self, client_updates, client_steps):
        """Aggregate normalized updates"""
        # Effective number of steps
        tau_eff = sum(client_steps) / len(client_steps)
        
        # Aggregate
        global_update = sum(
            tau_eff * update 
            for update in client_updates
        ) / len(client_updates)
        
        return global_weights + global_update
```

---

## 5. Per-FedAvg (Personalized FedAvg)

### **What It Is**
Learn a global model that can be quickly personalized for each facility.

### **How It Works**
```python
# Two-phase approach:

Phase 1: Federated Learning
- Learn global model (shared knowledge)

Phase 2: Personalization
- Each facility fine-tunes global model on local data
- Creates personalized model for that facility

Result:
- Global model: 80% accuracy (works for all)
- Personalized A: 92% accuracy (specialized for A)
- Personalized B: 90% accuracy (specialized for B)
- Personalized C: 88% accuracy (specialized for C)
```

### **Pros**
- ✅ Best accuracy for each facility
- ✅ Handles extreme heterogeneity
- ✅ Each facility gets custom model

### **Cons**
- ❌ Requires two-phase training
- ❌ More complex
- ❌ Each facility needs to store personalized model

### **When to Use**
- Facilities are very different
- Need best possible accuracy per facility
- Can afford two-phase training

### **Code Example**
```python
class PerFedAvgClient:
    """Personalized FedAvg client"""
    
    def __init__(self):
        self.global_model = create_model()
        self.personalized_model = None
    
    def train_global(self, global_weights):
        """Phase 1: Train global model"""
        self.global_model.set_weights(global_weights)
        self.global_model.fit(self.data, epochs=5)
        return self.global_model.get_weights()
    
    def personalize(self):
        """Phase 2: Create personalized model"""
        # Start from global model
        self.personalized_model = clone_model(self.global_model)
        
        # Fine-tune on local data (few epochs)
        self.personalized_model.fit(self.data, epochs=2)
        
        return self.personalized_model
    
    def predict(self, x):
        """Use personalized model for inference"""
        return self.personalized_model.predict(x)
```

---

## 6. Clustered Federated Learning

### **What It Is**
Group similar facilities and train separate models per cluster.

### **How It Works**
```python
# Step 1: Cluster facilities
Cluster 1 (Chemical plants):
- Facility A, D, G
- Train Model 1

Cluster 2 (Water treatment):
- Facility B, E, H
- Train Model 2

Cluster 3 (Power plants):
- Facility C, F, I
- Train Model 3

# Each cluster has its own FL process
```

### **Pros**
- ✅ Models specialized for facility types
- ✅ Better than single global model
- ✅ Handles systematic differences

### **Cons**
- ❌ Requires clustering (how to group?)
- ❌ Need enough facilities per cluster
- ❌ Less knowledge sharing across clusters

### **When to Use**
- Clear facility types (chemical, water, power)
- Many facilities (10+)
- Facilities within type are similar

### **Code Example**
```python
class ClusteredFL:
    """Clustered federated learning"""
    
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clusters = {}
        self.cluster_models = {}
    
    def cluster_facilities(self, facilities):
        """Cluster based on data statistics"""
        from sklearn.cluster import KMeans
        
        # Extract features for clustering
        features = []
        for facility in facilities:
            stats = {
                'mean_traffic': facility.data['packets_per_sec'].mean(),
                'std_traffic': facility.data['packets_per_sec'].std(),
                'attack_ratio': (facility.data['label'] > 0).mean(),
                # ... more features
            }
            features.append(list(stats.values()))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        labels = kmeans.fit_predict(features)
        
        # Group facilities
        for i, facility in enumerate(facilities):
            cluster_id = labels[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(facility)
        
        return self.clusters
    
    def train(self):
        """Train separate model per cluster"""
        for cluster_id, facilities in self.clusters.items():
            # Run FL within cluster
            model = create_model()
            
            for round in range(num_rounds):
                # Standard FL within cluster
                updates = [f.train(model) for f in facilities]
                model = aggregate(updates)
            
            self.cluster_models[cluster_id] = model
```

---

## 7. FedDyn (Federated Dynamic Regularization)

### **What It Is**
Dynamically adjust regularization based on client drift.

### **How It Works**
```python
# Similar to FedProx but adaptive
# Regularization strength changes based on how much client drifts

If client drifts a lot:
    → Increase regularization (pull back harder)

If client drifts a little:
    → Decrease regularization (allow more freedom)
```

### **Pros**
- ✅ Adaptive to heterogeneity level
- ✅ Better than fixed FedProx
- ✅ Theoretically sound

### **Cons**
- ❌ More complex than FedProx
- ❌ Requires tracking drift
- ❌ More hyperparameters

### **When to Use**
- Heterogeneity varies over time
- Need adaptive approach
- Have time for complex implementation

---

## 📊 Comparison for Your Use Case

### **Your Scenario:**
- 3 facilities (Chemical, Water, Power)
- Different network patterns (Modbus vs MQTT)
- Different attack distributions
- Different data volumes
- **Timeline: 4 days**

### **Recommendations:**

#### **Day 1-2: Start Simple**
```
1. FedAvg (baseline)
   - Time: 1 hour
   - Accuracy: ~68%
   - Purpose: Baseline

2. FedBN (quick win)
   - Time: +1 hour
   - Accuracy: ~78% (+10%)
   - Purpose: Handle distribution shift
```

#### **Day 2-3: Add FedProx**
```
3. FedProx (recommended)
   - Time: +2 hours
   - Accuracy: ~82% (+14%)
   - Purpose: Handle heterogeneity
   - μ = 0.01 (start here)
```

#### **Day 3-4: Optional Enhancements**
```
4. FedProx + FedBN (combine)
   - Time: +1 hour
   - Accuracy: ~85% (+17%)
   - Purpose: Best of both

5. Clustered FL (if time permits)
   - Time: +4 hours
   - Accuracy: ~87% (+19%)
   - Purpose: Specialized models
```

---

## 🎯 Decision Matrix

### **Choose FedAvg if:**
- ✅ Just testing FL basics
- ✅ All facilities have similar data
- ✅ Need quick baseline

### **Choose FedBN if:**
- ✅ Different traffic patterns (your case!)
- ✅ Model has BatchNorm layers
- ✅ Want quick improvement (1 hour)

### **Choose FedProx if:**
- ✅ Different attack distributions (your case!)
- ✅ Different data volumes (your case!)
- ✅ Want good balance of simplicity/performance

### **Choose SCAFFOLD if:**
- ✅ Very high heterogeneity
- ✅ Need best performance
- ✅ Have time (1 day implementation)

### **Choose FedNova if:**
- ✅ Different hardware speeds
- ✅ Clients train different amounts
- ✅ Combine with FedProx

### **Choose Per-FedAvg if:**
- ✅ Need personalized models
- ✅ Facilities are very different
- ✅ Can afford two-phase training

### **Choose Clustered FL if:**
- ✅ Clear facility types
- ✅ Many facilities (10+)
- ✅ Want specialized models

---

## 💡 Practical Recommendation for 4 Days

### **Day 1: Baseline**
```python
# FedAvg (1 hour)
accuracy = 68%
```

### **Day 2: Quick Win**
```python
# FedAvg + FedBN (2 hours total)
accuracy = 78% (+10%)
```

### **Day 3: Main Solution**
```python
# FedProx + FedBN (4 hours total)
accuracy = 85% (+17%)
```

### **Day 4: Polish**
```python
# Tune hyperparameters, document
# Final accuracy: 85-87%
```

---

## 📈 Expected Results Comparison

| Method | Avg Accuracy | Fairness Gap | Implementation Time |
|--------|--------------|--------------|---------------------|
| FedAvg | 68% | 27% | 1 hour |
| FedBN | 78% | 18% | 2 hours |
| FedProx | 82% | 8% | 3 hours |
| FedBN + FedProx | 85% | 6% | 4 hours |
| SCAFFOLD | 88% | 4% | 1 day |
| Clustered FL | 87% | 5% | 6 hours |

---

## 🚀 Quick Start Code

### **Option 1: FedBN (Easiest)**
```python
# Just exclude BatchNorm from aggregation
def get_shareable_weights(model):
    weights = []
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            weights.append(layer.get_weights())
    return weights
```

### **Option 2: FedProx (Recommended)**
```python
# Add proximal term to loss
loss = data_loss + (0.01/2) * ||w_local - w_global||²
```

### **Option 3: FedBN + FedProx (Best)**
```python
# Combine both approaches
# 1. Exclude BatchNorm from aggregation
# 2. Add proximal term to loss
```

---

## Summary

### **For Your 4-Day Deadline:**

**Best choice: FedProx + FedBN**
- ✅ 4 hours implementation
- ✅ +17% accuracy improvement
- ✅ Handles your heterogeneity
- ✅ Simple enough for deadline

**Alternative: Just FedProx**
- ✅ 3 hours implementation
- ✅ +14% accuracy improvement
- ✅ Simpler than combination

**Avoid for now:**
- ❌ SCAFFOLD (too complex, 1 day)
- ❌ Clustered FL (need more facilities)
- ❌ Per-FedAvg (two-phase training)

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Recommendation:** FedProx (or FedProx + FedBN if time permits)
