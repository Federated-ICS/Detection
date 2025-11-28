# Detection Module - File Organization

**Last Updated:** November 25, 2025  
**Status:** Organized for 4-day deadline

---

## 📁 Folder Structure

```
Detection/
├── README.md                    ⭐ Module overview
├── START_HERE.md                ⭐⭐⭐ READ THIS FIRST (4-day plan)
├── 4_DAY_PLAN.md                ⭐⭐⭐ Detailed 4-day roadmap
├── FILE_ORGANIZATION.md         📋 This file
├── requirements.txt             📦 Python dependencies
│
├── notebooks/                   📓 Jupyter notebooks
│   ├── preprocessing.ipynb      - Data preprocessing pipeline
│   └── train.ipynb              - CNN-LSTM model training
│
├── fl_code/                     🔧 Federated learning code
│   ├── fl_simple_example.py     ⭐⭐⭐ Working FL implementation
│   ├── prepare_fl_data_simple.py ⭐⭐ Data preparation script
│   └── simple_normalizer.py     ⭐⭐ Heterogeneity solution
│
└── docs/                        📚 Documentation
    ├── FEDERATED_LEARNING_GUIDE.md      - Complete FL guide
    ├── FL_QUICK_START.md                - Quick reference
    ├── FL_SUGGESTIONS.md                - Implementation tips
    ├── FL_SUMMARY.md                    - FL summary
    └── DATA_HETEROGENEITY_SOLUTION.md   - Heterogeneity handling
```

---

## 🎯 Quick Navigation

### **Starting Your 4-Day Journey?**
1. **`START_HERE.md`** ← Start here!
2. **`4_DAY_PLAN.md`** ← Your roadmap
3. **`fl_code/fl_simple_example.py`** ← Working code

### **Need to Understand FL?**
1. **`docs/FL_QUICK_START.md`** ← Quick reference
2. **`docs/FEDERATED_LEARNING_GUIDE.md`** ← Complete guide
3. **`docs/FL_SUMMARY.md`** ← Summary

### **Working on Heterogeneity?**
1. **`docs/DATA_HETEROGENEITY_SOLUTION.md`** ← Solutions
2. **`fl_code/simple_normalizer.py`** ← Implementation

### **Training Detection Model?**
1. **`notebooks/preprocessing.ipynb`** ← Data prep
2. **`notebooks/train.ipynb`** ← Model training
3. **`README.md`** ← Module overview

---

## 📋 File Descriptions

### Root Files (Priority Order)

| File | Priority | Purpose | When to Use |
|------|----------|---------|-------------|
| `START_HERE.md` | ⭐⭐⭐ | 4-day quick start | First thing to read |
| `4_DAY_PLAN.md` | ⭐⭐⭐ | Detailed daily plan | Planning your work |
| `README.md` | ⭐⭐ | Module overview | Understanding Detection |
| `FILE_ORGANIZATION.md` | ⭐ | This file | Finding files |
| `requirements.txt` | ⭐ | Dependencies | Installation |

---

### `notebooks/` - Jupyter Notebooks

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `preprocessing.ipynb` | Data preprocessing | `DNN-EdgeIIoT-dataset.csv` | `X_train.csv`, `y_train.csv` |
| `train.ipynb` | Model training | Preprocessed CSVs | `best_multiclass_cnn_lstm_model.h5` |

**When to use:**
- Training standalone CNN-LSTM model
- Understanding data preprocessing
- Experimenting with model architecture

---

### `fl_code/` - Federated Learning Code

| File | Priority | Purpose | Lines |
|------|----------|---------|-------|
| `fl_simple_example.py` | ⭐⭐⭐ | Complete FL implementation | ~300 |
| `prepare_fl_data_simple.py` | ⭐⭐ | Split data for 3 facilities | ~100 |
| `simple_normalizer.py` | ⭐⭐ | Per-facility normalization | ~150 |

**Usage:**
```bash
# Prepare data
python fl_code/prepare_fl_data_simple.py --samples 10000

# Run FL server
python fl_code/fl_simple_example.py server 5 3

# Run FL clients
python fl_code/fl_simple_example.py client facility_a
python fl_code/fl_simple_example.py client facility_b
python fl_code/fl_simple_example.py client facility_c
```

---

### `docs/` - Documentation

| File | Pages | Purpose | Audience |
|------|-------|---------|----------|
| `FEDERATED_LEARNING_GUIDE.md` | ~30 | Complete FL implementation | Developers |
| `FL_QUICK_START.md` | ~15 | Quick reference & tips | Everyone |
| `FL_SUGGESTIONS.md` | ~20 | Best practices | Implementers |
| `FL_SUMMARY.md` | ~10 | High-level overview | Beginners |
| `DATA_HETEROGENEITY_SOLUTION.md` | ~25 | Heterogeneity handling | ML Engineers |

**Reading order:**
1. Beginner: `FL_SUMMARY.md` → `FL_QUICK_START.md`
2. Implementer: `FEDERATED_LEARNING_GUIDE.md` → `FL_SUGGESTIONS.md`
3. Advanced: `DATA_HETEROGENEITY_SOLUTION.md`

---

## 🚀 Common Workflows

### Workflow 1: Train Standalone Model
```bash
# 1. Preprocess data
jupyter notebook notebooks/preprocessing.ipynb

# 2. Train model
jupyter notebook notebooks/train.ipynb

# Output: best_multiclass_cnn_lstm_model.h5
```

---

### Workflow 2: Run FL Demo (4-Day Plan)
```bash
# Day 1: Get FL working
pip install flwr
python fl_code/prepare_fl_data_simple.py --samples 10000
python fl_code/fl_simple_example.py server 5 3
python fl_code/fl_simple_example.py client facility_a

# Day 2: Add normalization
# (Edit fl_simple_example.py to use simple_normalizer.py)

# Day 3: Demo scenario
# (Create specialized datasets)

# Day 4: Polish
# (Create demo script)
```

---

### Workflow 3: Understand FL Concepts
```bash
# 1. Read summary
cat docs/FL_SUMMARY.md

# 2. Read quick start
cat docs/FL_QUICK_START.md

# 3. Try example
python fl_code/fl_simple_example.py server 1 3
```

---

### Workflow 4: Handle Heterogeneity
```bash
# 1. Read solution guide
cat docs/DATA_HETEROGENEITY_SOLUTION.md

# 2. Use normalizer
python fl_code/simple_normalizer.py  # Test

# 3. Integrate with FL
# (Update fl_simple_example.py)
```

---

## 📊 File Sizes & Complexity

| File | Size | Complexity | Time to Read |
|------|------|------------|--------------|
| `START_HERE.md` | 5 KB | ⭐ Easy | 5 min |
| `4_DAY_PLAN.md` | 15 KB | ⭐⭐ Medium | 15 min |
| `fl_simple_example.py` | 10 KB | ⭐⭐ Medium | 20 min |
| `FEDERATED_LEARNING_GUIDE.md` | 40 KB | ⭐⭐⭐ Hard | 60 min |
| `DATA_HETEROGENEITY_SOLUTION.md` | 35 KB | ⭐⭐⭐ Hard | 45 min |

---

## 🎯 Files by Use Case

### Use Case: "I have 4 days to demo FL"
**Read:**
1. `START_HERE.md` (5 min)
2. `4_DAY_PLAN.md` (15 min)

**Use:**
1. `fl_code/fl_simple_example.py`
2. `fl_code/prepare_fl_data_simple.py`

**Reference:**
1. `docs/FL_QUICK_START.md`

---

### Use Case: "I want to understand FL deeply"
**Read:**
1. `docs/FL_SUMMARY.md` (10 min)
2. `docs/FEDERATED_LEARNING_GUIDE.md` (60 min)
3. `docs/DATA_HETEROGENEITY_SOLUTION.md` (45 min)

**Try:**
1. `fl_code/fl_simple_example.py`

---

### Use Case: "I need to handle heterogeneous data"
**Read:**
1. `docs/DATA_HETEROGENEITY_SOLUTION.md` (45 min)

**Use:**
1. `fl_code/simple_normalizer.py`

**Reference:**
1. `docs/FL_SUGGESTIONS.md`

---

### Use Case: "I want to train detection model"
**Read:**
1. `README.md` (10 min)

**Use:**
1. `notebooks/preprocessing.ipynb`
2. `notebooks/train.ipynb`

---

## 🔍 Finding Specific Information

### "How do I install dependencies?"
→ `requirements.txt` or `README.md`

### "How do I run FL?"
→ `START_HERE.md` or `docs/FL_QUICK_START.md`

### "How do I handle different facilities?"
→ `docs/DATA_HETEROGENEITY_SOLUTION.md`

### "What's the CNN-LSTM architecture?"
→ `README.md` or `notebooks/train.ipynb`

### "How do I prepare data?"
→ `notebooks/preprocessing.ipynb` or `fl_code/prepare_fl_data_simple.py`

### "What if I'm stuck?"
→ `docs/FL_QUICK_START.md` (Troubleshooting section)

### "How do I prove knowledge transfer?"
→ `4_DAY_PLAN.md` (Day 3)

---

## 📦 Generated Files (Not in Git)

These files are created when you run the code:

```
Detection/
├── fl_data/                     # Generated by prepare_fl_data_simple.py
│   ├── facility_a/
│   │   ├── X_train.csv
│   │   ├── y_train.csv
│   │   └── normalizer.pkl
│   ├── facility_b/
│   └── facility_c/
│
├── X_train.csv                  # Generated by preprocessing.ipynb
├── y_train.csv
├── X_val.csv
├── y_val.csv
├── X_test.csv
├── y_test.csv
│
├── best_multiclass_cnn_lstm_model.h5  # Generated by train.ipynb
├── best_binary_cnn_lstm_model.h5
│
└── *.log                        # Generated by FL runs
```

---

## 🗂️ File Dependencies

```
START_HERE.md
    ↓ references
4_DAY_PLAN.md
    ↓ uses
fl_code/fl_simple_example.py
    ↓ uses
fl_code/prepare_fl_data_simple.py
    ↓ creates
fl_data/facility_*/X_train.csv

fl_code/fl_simple_example.py
    ↓ can use
fl_code/simple_normalizer.py
    ↓ explained in
docs/DATA_HETEROGENEITY_SOLUTION.md

notebooks/preprocessing.ipynb
    ↓ creates
X_train.csv, y_train.csv
    ↓ used by
notebooks/train.ipynb
    ↓ creates
best_multiclass_cnn_lstm_model.h5
```

---

## 🎓 Learning Path

### Beginner (Day 1)
1. Read `START_HERE.md`
2. Read `4_DAY_PLAN.md` (Day 1 only)
3. Run `fl_code/fl_simple_example.py`
4. Reference `docs/FL_QUICK_START.md` if stuck

### Intermediate (Day 2-3)
1. Read `docs/FEDERATED_LEARNING_GUIDE.md`
2. Read `docs/DATA_HETEROGENEITY_SOLUTION.md`
3. Modify `fl_code/fl_simple_example.py`
4. Create demo scenario

### Advanced (Day 4+)
1. Read `docs/FL_SUGGESTIONS.md`
2. Implement advanced features
3. Optimize performance
4. Deploy to production

---

## 📝 Maintenance

### Adding New Files
- **Code:** Add to `fl_code/`
- **Documentation:** Add to `docs/`
- **Notebooks:** Add to `notebooks/`
- **Update:** This file (`FILE_ORGANIZATION.md`)

### Updating Documentation
1. Update the specific doc file
2. Update `README.md` if needed
3. Update this file if structure changes
4. Update `START_HERE.md` if workflow changes

---

## 🔗 External Resources

### Flower (FL Framework)
- Docs: https://flower.dev/docs/
- Examples: https://github.com/adap/flower/tree/main/examples
- Slack: https://flower.dev/join-slack

### TensorFlow/Keras
- TensorFlow: https://www.tensorflow.org/guide
- Keras: https://keras.io/guides/

### Dataset
- DNN-EdgeIIoT: https://ieee-dataport.org/documents/edge-iiotset

---

## ✅ Quick Checklist

### Before Starting
- [ ] Read `START_HERE.md`
- [ ] Read `4_DAY_PLAN.md`
- [ ] Install dependencies (`requirements.txt`)
- [ ] Have dataset ready

### Day 1
- [ ] Run `fl_code/prepare_fl_data_simple.py`
- [ ] Run `fl_code/fl_simple_example.py`
- [ ] Verify FL works

### Day 2
- [ ] Integrate `fl_code/simple_normalizer.py`
- [ ] Test with normalization
- [ ] Compare results

### Day 3
- [ ] Create demo scenario
- [ ] Test knowledge transfer
- [ ] Document results

### Day 4
- [ ] Create demo script
- [ ] Write README
- [ ] Test end-to-end
- [ ] Prepare presentation

---

## 📞 Getting Help

### File-Specific Issues
- **FL not working:** Check `docs/FL_QUICK_START.md` (Troubleshooting)
- **Data issues:** Check `notebooks/preprocessing.ipynb`
- **Model issues:** Check `notebooks/train.ipynb`
- **Heterogeneity:** Check `docs/DATA_HETEROGENEITY_SOLUTION.md`

### General Issues
- **Installation:** Check `requirements.txt` and `README.md`
- **Concepts:** Check `docs/FL_SUMMARY.md`
- **Implementation:** Check `docs/FEDERATED_LEARNING_GUIDE.md`

---

## 🎯 Summary

**Essential Files (Must Use):**
1. `START_HERE.md` - Your starting point
2. `4_DAY_PLAN.md` - Your roadmap
3. `fl_code/fl_simple_example.py` - Working FL code

**Reference Files (Use When Needed):**
- `docs/FL_QUICK_START.md` - Quick help
- `docs/FEDERATED_LEARNING_GUIDE.md` - Deep dive
- `docs/DATA_HETEROGENEITY_SOLUTION.md` - Advanced topics

**Everything else:** Supporting documentation and code

---

**Last Updated:** November 25, 2025  
**Status:** Organized and ready for 4-day sprint  
**Next:** Read `START_HERE.md` and begin!
