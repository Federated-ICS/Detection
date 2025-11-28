# Project Structure
## Federated Network-Based ICS Threat Detection System

**Last Updated:** November 28, 2025  
**Status:** Organized and Production-Ready

---

## 📁 Directory Structure

```
Detection/
│
├── README.md                          # Main project overview
├── requirements.txt                   # Python dependencies
├── PROJECT_STRUCTURE.md              # This file
│
├── documentation/                     # 📚 All documentation
│   ├── guides/                       # Getting started guides
│   │   ├── START_HERE.md            # ⭐ Start here for 4-day plan
│   │   ├── 4_DAY_PLAN.md            # Detailed 4-day roadmap
│   │   └── FILE_ORGANIZATION.md     # Legacy file organization
│   │
│   ├── research/                     # Research documentation
│   │   ├── Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md
│   │   └── paper.txt                # Research notes
│   │
│   └── federated_learning/          # FL-specific documentation
│       ├── FL_HETEROGENEOUS_LABELS_SOLUTION.md
│       ├── FL_NEW_THREAT_DISCOVERY.md
│       └── FL_LEARNING_RESOURCES.md
│
├── docs/                             # Original documentation
│   ├── DATA_HETEROGENEITY_SOLUTION.md
│   ├── FEDERATED_LEARNING_GUIDE.md
│   ├── FL_ALTERNATIVES.md
│   ├── FL_QUICK_START.md
│   ├── FL_SUGGESTIONS.md
│   └── FL_SUMMARY.md
│
├── fl_code/                          # 🔧 Federated learning code
│   ├── fl_simple_example.py         # ⭐ Working FL implementation
│   ├── prepare_fl_data_simple.py    # Data preparation
│   └── simple_normalizer.py         # Heterogeneity handling
│
├── notebooks/                        # 📓 Jupyter notebooks
│   ├── detection-13.ipynb           # Filtered dataset experiment
│   ├── preprocessing.ipynb          # Data preprocessing
│   └── train.ipynb                  # Model training
│
└── models/                           # 💾 Trained models
    └── saved_models/
        └── EdgeIIoT_filtered_model/ # Filtered dataset model
            ├── best_multiclass_cnn_lstm_model.h5
            ├── cnn_lstm_filtered_final_model.keras
            ├── label_encoder_filtered.pkl
            ├── scaler_filtered.pkl
            ├── model_results_filtered.pkl
            ├── multiclass_confusion_matrix.png
            └── multiclass_training_history.png
```

---

## 🚀 Quick Navigation

### Getting Started
1. **New to the project?** → `documentation/guides/START_HERE.md`
2. **4-day deadline?** → `documentation/guides/4_DAY_PLAN.md`
3. **Want to understand FL?** → `docs/FL_SUMMARY.md`

### Research & Methodology
1. **Research summary** → `documentation/research/Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md`
2. **Full FL guide** → `docs/FEDERATED_LEARNING_GUIDE.md`
3. **Data heterogeneity** → `docs/DATA_HETEROGENEITY_SOLUTION.md`

### Federated Learning Challenges
1. **Different attack types per facility** → `documentation/federated_learning/FL_HETEROGENEOUS_LABELS_SOLUTION.md`
2. **New threat discovery** → `documentation/federated_learning/FL_NEW_THREAT_DISCOVERY.md`
3. **Learning resources** → `documentation/federated_learning/FL_LEARNING_RESOURCES.md`

### Implementation
1. **Run FL demo** → `fl_code/fl_simple_example.py`
2. **Prepare data** → `fl_code/prepare_fl_data_simple.py`
3. **Train model** → `notebooks/train.ipynb`

---

## 📋 File Descriptions

### Root Files

| File | Purpose | Priority |
|------|---------|----------|
| `README.md` | Project overview and main documentation | ⭐⭐⭐ |
| `requirements.txt` | Python dependencies | ⭐⭐⭐ |
| `PROJECT_STRUCTURE.md` | This file - navigation guide | ⭐⭐ |

### Documentation Structure

#### `/documentation/guides/` - Getting Started
- **START_HERE.md** - Your entry point, 4-day quick start
- **4_DAY_PLAN.md** - Detailed daily plan with commands
- **FILE_ORGANIZATION.md** - Legacy organization reference

#### `/documentation/research/` - Research Materials
- **Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md** - Complete research methodology
- **paper.txt** - Research notes and references

#### `/documentation/federated_learning/` - FL Solutions
- **FL_HETEROGENEOUS_LABELS_SOLUTION.md** - Handle different attack types per facility
- **FL_NEW_THREAT_DISCOVERY.md** - Dynamic threat discovery and model expansion
- **FL_LEARNING_RESOURCES.md** - 34 papers, tutorials, and learning materials

### Original Documentation (`/docs/`)

| File | Purpose | When to Use |
|------|---------|-------------|
| `FL_SUMMARY.md` | High-level FL overview | First-time learning |
| `FL_QUICK_START.md` | Quick reference guide | During implementation |
| `FEDERATED_LEARNING_GUIDE.md` | Complete FL implementation | Deep dive |
| `DATA_HETEROGENEITY_SOLUTION.md` | Handle heterogeneous data | Advanced scenarios |
| `FL_SUGGESTIONS.md` | Best practices | Planning phase |
| `FL_ALTERNATIVES.md` | Alternative approaches | Exploring options |

### Code (`/fl_code/`)

| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `fl_simple_example.py` | ~300 | Complete FL implementation | ⭐⭐⭐ |
| `prepare_fl_data_simple.py` | ~100 | Split data for facilities | ⭐⭐ |
| `simple_normalizer.py` | ~150 | Per-facility normalization | ⭐⭐ |

### Notebooks (`/notebooks/`)

| File | Purpose | Dataset |
|------|---------|---------|
| `preprocessing.ipynb` | Data preprocessing pipeline | Full (2.2M samples) |
| `train.ipynb` | CNN-LSTM model training | Full (15 classes) |
| `detection-13.ipynb` | Filtered dataset experiment | Filtered (11 classes) |

### Models (`/models/saved_models/`)

**EdgeIIoT_filtered_model/** - Trained on 11 attack types
- Model files: `.h5`, `.keras`
- Preprocessing: `label_encoder_filtered.pkl`, `scaler_filtered.pkl`
- Results: `model_results_filtered.pkl`
- Visualizations: `.png` files

---

## 🎯 Common Tasks

### Task 1: Run FL Demo
```bash
# 1. Prepare data
cd fl_code
python prepare_fl_data_simple.py --samples 10000

# 2. Start server (Terminal 1)
python fl_simple_example.py server 5 3

# 3. Start clients (Terminals 2-4)
python fl_simple_example.py client facility_a
python fl_simple_example.py client facility_b
python fl_simple_example.py client facility_c
```

### Task 2: Train Standalone Model
```bash
# 1. Open preprocessing notebook
jupyter notebook notebooks/preprocessing.ipynb

# 2. Open training notebook
jupyter notebook notebooks/train.ipynb
```

### Task 3: Understand FL Concepts
```bash
# Read in order:
1. documentation/guides/START_HERE.md
2. docs/FL_SUMMARY.md
3. docs/FL_QUICK_START.md
4. docs/FEDERATED_LEARNING_GUIDE.md
```

### Task 4: Handle Different Attack Types
```bash
# Read solution:
documentation/federated_learning/FL_HETEROGENEOUS_LABELS_SOLUTION.md

# Implement global label space approach
```

### Task 5: Handle New Threat Discovery
```bash
# Read solution:
documentation/federated_learning/FL_NEW_THREAT_DISCOVERY.md

# Implement dynamic model expansion
```

---

## 📊 Documentation Map

### By Experience Level

**Beginner (New to FL):**
1. `documentation/guides/START_HERE.md`
2. `docs/FL_SUMMARY.md`
3. `fl_code/fl_simple_example.py`

**Intermediate (Implementing FL):**
1. `docs/FEDERATED_LEARNING_GUIDE.md`
2. `docs/FL_QUICK_START.md`
3. `documentation/federated_learning/FL_HETEROGENEOUS_LABELS_SOLUTION.md`

**Advanced (Research & Production):**
1. `documentation/research/Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md`
2. `documentation/federated_learning/FL_NEW_THREAT_DISCOVERY.md`
3. `docs/DATA_HETEROGENEITY_SOLUTION.md`

### By Problem

| Problem | Solution Document |
|---------|------------------|
| Getting started | `documentation/guides/START_HERE.md` |
| Different attack types | `documentation/federated_learning/FL_HETEROGENEOUS_LABELS_SOLUTION.md` |
| New threats discovered | `documentation/federated_learning/FL_NEW_THREAT_DISCOVERY.md` |
| Data heterogeneity | `docs/DATA_HETEROGENEITY_SOLUTION.md` |
| Learning FL concepts | `documentation/federated_learning/FL_LEARNING_RESOURCES.md` |
| Quick reference | `docs/FL_QUICK_START.md` |

---

## 🔄 Workflow Paths

### Path 1: Quick Demo (4 hours)
```
START_HERE.md → prepare_fl_data_simple.py → fl_simple_example.py → Done!
```

### Path 2: Full Implementation (2 weeks)
```
FL_SUMMARY.md → FEDERATED_LEARNING_GUIDE.md → 
FL_HETEROGENEOUS_LABELS_SOLUTION.md → Implementation → Testing
```

### Path 3: Research & Paper (3 months)
```
Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md → 
FL_LEARNING_RESOURCES.md → Read papers → 
Implement solutions → Write paper → Submit
```

---

## 📦 Generated Files (Not in Git)

These files are created when you run the code:

```
fl_data/                              # Generated by prepare_fl_data_simple.py
├── facility_a/
│   ├── X_train.csv
│   ├── y_train.csv
│   └── normalizer.pkl
├── facility_b/
└── facility_c/

X_train.csv, y_train.csv, ...        # Generated by preprocessing.ipynb
best_multiclass_cnn_lstm_model.h5    # Generated by train.ipynb
*.log                                 # Generated by FL runs
```

---

## 🎓 Learning Path

### Week 1: Foundations
- [ ] Read `START_HERE.md`
- [ ] Read `FL_SUMMARY.md`
- [ ] Run `fl_simple_example.py`
- [ ] Understand FedAvg algorithm

### Week 2: Implementation
- [ ] Read `FEDERATED_LEARNING_GUIDE.md`
- [ ] Implement global label space
- [ ] Test with 3 facilities
- [ ] Handle heterogeneity

### Week 3: Advanced Topics
- [ ] Read `FL_NEW_THREAT_DISCOVERY.md`
- [ ] Implement dynamic expansion
- [ ] Read research papers
- [ ] Test knowledge transfer

### Week 4: Production
- [ ] Add differential privacy
- [ ] Deploy with Docker
- [ ] Create monitoring
- [ ] Write documentation

---

## 🔍 Search Guide

### "I want to..."

**...understand FL basics**
→ `docs/FL_SUMMARY.md`

**...run a quick demo**
→ `documentation/guides/START_HERE.md`

**...handle different attack types**
→ `documentation/federated_learning/FL_HETEROGENEOUS_LABELS_SOLUTION.md`

**...handle new threats**
→ `documentation/federated_learning/FL_NEW_THREAT_DISCOVERY.md`

**...learn from papers**
→ `documentation/federated_learning/FL_LEARNING_RESOURCES.md`

**...implement full FL system**
→ `docs/FEDERATED_LEARNING_GUIDE.md`

**...understand the research**
→ `documentation/research/Detection_model_RESEARCH_METHODOLOGY_SUMMARY.md`

**...train the model**
→ `notebooks/train.ipynb`

---

## 📞 Getting Help

### Documentation Issues
1. Check `PROJECT_STRUCTURE.md` (this file)
2. Check `documentation/guides/FILE_ORGANIZATION.md`
3. Check `README.md`

### Implementation Issues
1. Check `docs/FL_QUICK_START.md` (Troubleshooting section)
2. Check Flower docs: https://flower.dev/docs/
3. Check Flower Slack: https://flower.dev/join-slack

### Research Questions
1. Check `documentation/federated_learning/FL_LEARNING_RESOURCES.md`
2. Read relevant papers
3. Ask on research forums

---

## ✅ Organization Checklist

- [x] Created clear directory structure
- [x] Moved files to appropriate locations
- [x] Created navigation guide (this file)
- [x] Documented all files and purposes
- [x] Created learning paths
- [x] Added quick reference sections
- [x] Included troubleshooting guides

---

## 🎯 Next Steps

1. **Read this file** to understand the structure
2. **Navigate to** `documentation/guides/START_HERE.md`
3. **Follow** the 4-day plan or your chosen path
4. **Refer back** to this file when you need to find something

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** Complete and Organized  
**Total Files:** 25+ documentation files, 3 code files, 3 notebooks
