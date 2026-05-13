# CNN-LSTM Intrusion Detection System for IIoT Networks

This project implements a complete preprocessing and deep learning pipeline for intrusion detection in Industrial Internet of Things (IIoT) environments using the **Edge-IIoT dataset**.
The system combines feature engineering, imbalance handling, normalization, and a **CNN-LSTM hybrid architecture** for binary and multiclass attack classification.

---

## Project Structure

```bash
.
├── preprocessing.ipynb     # Data preprocessing pipeline
├── train.ipynb             # CNN-LSTM training and evaluation
├── DNN-EdgeIIoT-dataset.csv
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
├── y_test.csv
└── README.md
```

---

# Dataset

The project uses the **Edge-IIoT Dataset**, which contains normal and malicious network traffic generated in Industrial IoT environments.

The dataset includes:

* Network traffic features
* Multiple attack categories
* Normal traffic samples
* Mixed numerical and categorical attributes

---

# Preprocessing Pipeline

The notebook `preprocessing.ipynb` performs all preprocessing operations required before training.

## Steps Included

### 1. Dataset Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv('./../DNN-EdgeIIoT-dataset.csv')
```

---

### 2. Feature Encoding

Categorical features are converted into numerical representations using:

* `LabelEncoder`
* Feature transformation methods

---

### 3. Feature Selection

The project uses **Mutual Information (MI)** to identify the most relevant features.

Benefits:

* Reduces dimensionality
* Improves training speed
* Removes noisy features

---

### 4. Class Balancing with SMOTE

The dataset imbalance is handled using:

```python
SMOTE()
```

This generates synthetic samples for minority classes.

Visualization is also included to compare:

* Original distribution
* Balanced distribution

---

### 5. Feature Normalization

Features are standardized using:

```python
StandardScaler()
```

This ensures:

* Mean = 0
* Standard deviation = 1

---

### 6. Dataset Splitting

The data is divided into:

* Training set
* Validation set
* Test set

---

# Model Architecture

The notebook `train.ipynb` implements a hybrid **CNN-LSTM** deep learning model.

Two modes are supported:

* Binary Classification
* Multiclass Classification

---

## CNN-LSTM Overview

### CNN Layers

Used for:

* Spatial feature extraction
* Learning local traffic patterns

### LSTM Layers

Used for:

* Temporal dependency learning
* Sequential behavior analysis

---

## Binary Classification Model

Detects:

* Normal traffic
* Attack traffic

---

## Multiclass Classification Model

Classifies multiple attack categories individually.

---

# Training Process

The training pipeline includes:

* Data loading
* Feature scaling
* Label encoding
* Model compilation
* Training with validation monitoring

Loss functions:

* Binary Crossentropy
* Categorical Crossentropy

Optimizers:

* Adam Optimizer

Metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

# Evaluation

The project evaluates the trained model using:

* Classification Report
* Confusion Matrix
* Accuracy Curves
* Loss Curves

Visualization functions are included to monitor training history.

---

# Saved Outputs

After training, the notebook saves:

```bash
cnn_lstm_final_model.h5
```

Additional saved artifacts may include:

* Encoders
* Scalers
* Feature selection outputs

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Imbalanced-learn (SMOTE)

---

# How to Run

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn
```

---

## 2. Run Preprocessing

Open and execute:

```bash
preprocessing.ipynb
```

This generates the processed datasets.

---

## 3. Train the Model

Run:

```bash
train.ipynb
```

The notebook will:

* Train the CNN-LSTM model
* Evaluate performance
* Save the final model

---

# Applications

This project can be applied in:

* Industrial IoT security
* Smart factories
* Network intrusion detection systems
* Cybersecurity research
* Edge AI security systems

---


