# Quantum Machine Learning (QML) for Blockchain Chain Analysis

A hybrid **Quantum-Classical pipeline** for detecting fraudulent Ethereum wallet activity, built around a **Quantum Support Vector Machine (QSVM)** implemented in IBM Qiskit. This project targets the pseudonymity problem in public blockchains — where Sybil identities, wallet-hopping, and mixing services obscure illicit value transfer — by combining classical preprocessing with quantum kernel methods.

> **Status:** Preprocessing, feature engineering, and PCA-based exploratory analysis are complete. QSVM training, quantum kernel estimation, and comparative evaluation are in progress (see [Roadmap](#roadmap--future-work) below).

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Completed Work](#completed-work)
  - [Data Cleaning](#1-data-cleaning)
  - [Feature Selection](#2-feature-selection)
  - [Normalization](#3-normalization)
  - [PCA Dimensionality Reduction](#4-pca-dimensionality-reduction)
  - [Visualization & Findings](#5-visualization--findings)
- [Roadmap / Future Work](#roadmap--future-work)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Results Snapshot](#results-snapshot)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [License](#license)

---

## Overview

Public blockchains like Ethereum are transparent but pseudonymous — every transaction is visible, but addresses carry no inherent link to real-world identities. **Chain analysis** attempts to recover that link, or at least flag suspicious behavior, using only on-chain signals.

Classical detectors (Random Forest, RBF-kernel SVM, GCNs, RNNs) struggle with:

- **High-dimensional, sparse behavioral features** that weaken margin-based classifiers
- **Severe class imbalance** (illicit wallets are a small minority)
- **Adversarial drift** — laundering tactics evolve faster than static models can adapt

This project explores whether **Quantum Machine Learning (QML)** — specifically quantum kernel methods — can recover fraud signatures that are indistinguishable to classical models, by mapping wallet features into an exponentially large quantum Hilbert space.

## Motivation

Quantum kernel methods leverage superposition and entanglement to represent data in feature spaces that have no efficient classical analogue. Prior theoretical and empirical work (Rebentrost et al., Schuld & Killoran, Havlíček et al.) suggests quantum-enhanced feature spaces can improve class separability on small-to-medium classification benchmarks. This project applies that idea concretely to **Ethereum fraud detection**, an area where, to the best of our knowledge, no prior study has deployed an explicit QSVM with quantum kernel estimation on a real labeled dataset and benchmarked it against strong classical baselines end-to-end.

## Pipeline Architecture

The system follows a **six-stage hybrid quantum-classical pipeline**:

```
Stage 1: Data Source
      ↓
Stage 2: Feature Engineering
      ↓
Stage 3: Preprocessing (cleaning, normalization, PCA)
      ↓
Stage 4: Quantum Encoding            ← [Future Work]
      ↓
Stage 5: QSVM Training               ← [Future Work]
      ↓
Stage 6: Evaluation                  ← [Future Work]
```

**Stages 1–3 are complete.** Stages 4–6 are the primary focus of ongoing development.

## Dataset

- **Source:** Kaggle Ethereum Fraud Detection Dataset
- **Size:** ~9,841 labeled wallet records
- **Labels:** Binary (`1` = illicit, `0` = legitimate)
- **Class distribution:** ~8,836 legitimate (89.8%) vs. ~1,005 fraudulent (10.2%) — a strongly imbalanced regime

## Completed Work

### 1. Data Cleaning
Missing entries were identified and removed; duplicate rows were dropped to eliminate redundancy that could bias downstream learning. The result is a clean wallet dataset ready for feature engineering.

### 2. Feature Selection
Three behaviorally informative features were selected for the current phase:

| Feature | Description |
|---|---|
| `avg_val_received` | Average ETH value received per transaction |
| `avg_val_sent` | Average ETH value sent per transaction |
| `total_ether_sent` | Cumulative ETH sent from the wallet |

Additional features (gas consumed, gas price, inter-transaction time gap) are retained and reserved for the full QSVM stage.

### 3. Normalization
All selected continuous features were rescaled to `[0, 1]` using Min-Max normalization:

```
x_norm = (x - x_min) / (x_max - x_min)
```

This prevents features with large absolute ranges from dominating distance computations in both PCA and the quantum kernel matrix.

### 4. PCA Dimensionality Reduction
Principal Component Analysis was applied to compress the normalized feature vectors to **2 components** for exploratory visualization:

```
z = Wᵀx
```

where `W` holds the top-N eigenvectors of the training covariance matrix. For the full QSVM phase, `N ∈ {4, 6, 8}` will be evaluated to match the qubit budget of the quantum feature map.

### 5. Visualization & Findings
A 2-D scatter plot of PC1 vs. PC2 was generated with class labels overlaid. Key observations:

- **Majority cluster:** Both classes form a dense cluster near the origin, reflecting high behavioral overlap in raw feature space.
- **Partial separation of fraud points:** A subset of fraudulent records appears slightly displaced along the first principal component — suggesting a detectable statistical signature even in low dimensions.
- **Overlap due to class imbalance:** The dominant legitimate class visually masks the minority fraud cluster, consistent with known imbalanced-data challenges.
- **Implication for quantum encoding:** The partial separability observed motivates lifting features into a higher-dimensional quantum Hilbert space, where quantum kernel methods are theoretically expected to increase the margin between overlapping classes.

These findings validate the pipeline design and support the hypothesis that quantum kernel-based separation is a meaningful direction for this dataset.

---

## Roadmap / Future Work

The following stages are planned to complete the hybrid pipeline:

### Quantum Encoding & QSVM Implementation
- Full QSVM implementation using **IBM Qiskit** and **PennyLane**.
- **Angle encoding** via parameterized `Rx`/`Ry` rotations as the primary feature map:

  ```
  |ψ(z)⟩ = ⊗ᵢ Rx(zᵢ) Ry(zᵢ) |0⟩
  ```

- Alternative feature maps to explore: `ZZFeatureMap`, `PauliFeatureMap`.

### Quantum Kernel Estimation
- Construct the quantum kernel (Gram) matrix by executing the circuit `U†(zⱼ)U(zᵢ)|0⟩^⊗N` on the **Qiskit Aer statevector simulator**, reading the probability of the all-zero outcome:

  ```
  K(zᵢ, zⱼ) = |⟨ψ(zᵢ)|ψ(zⱼ)⟩|²
  ```

- Sweep qubit count `N ∈ {4, 6, 8}`, matching the PCA component count accordingly.
- Feed the resulting Gram matrix into `sklearn.svm.SVC(kernel='precomputed')`.

### Class Imbalance Handling
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) exclusively on the training fold to counteract the ~9:1 class imbalance.
- Validation and test folds remain free of synthetic samples to avoid optimistic bias.

### Comparative Evaluation
Benchmark the QSVM against two classical baselines under identical preprocessing:

- **RBF-kernel SVM** (`C=1.0`, `gamma='scale'`)
- **Random Forest** (100 estimators, `class_weight='balanced'`)

Metrics: accuracy, precision, recall, F1-score, and AUC. **Fraud-class recall** is the primary operational metric for chain-analysis deployment.

### Real Hardware Deployment
- Validate simulation results on real **IBM Quantum** superconducting backends (5- and 7-qubit devices).
- Characterize the effect of gate noise, measurement error, and decoherence on kernel fidelity.

### Scalability
- Investigate quantum kernel approximation techniques (importance sampling, Nyström-style methods) to address the `O(m²)` kernel construction cost for larger transaction datasets.

### Real-Time Monitoring
- Integrate with live blockchain data via **Web3.py** and **Infura APIs** to enable real-time fraud flagging on newly confirmed Ethereum transactions.

---

## Tech Stack

- **Language:** Python
- **Classical ML:** scikit-learn (PCA, SVM, Random Forest, SMOTE)
- **Quantum Computing:** IBM Qiskit, PennyLane
- **Data Handling:** pandas, NumPy
- **Visualization:** Matplotlib
- **Planned Integration:** Web3.py, Infura API

## Project Structure

```
Quantum-Authentication-System/
├── data/                   # Raw and processed datasets
├── notebooks/              # Exploratory analysis, PCA visualization
├── src/
│   ├── preprocessing/      # Cleaning, feature selection, normalization
│   ├── pca/                # Dimensionality reduction
│   ├── quantum/            # Quantum encoding & kernel estimation (planned)
│   ├── classical/          # Baseline classical models
│   └── evaluation/         # Metrics & comparative evaluation (planned)
├── results/                # Figures, plots, metrics outputs
├── requirements.txt
└── README.md
```

*(Adjust this structure to match your actual repository layout.)*

## Getting Started

```bash
# Clone the repository
git clone https://github.com/arpitkumar2805/Quantum-Authentication-System.git
cd Quantum-Authentication-System

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Preprocessing Pipeline

```bash
python src/preprocessing/clean_data.py
python src/pca/reduce_dimensions.py
```

*(Update these commands to match your actual scripts.)*





