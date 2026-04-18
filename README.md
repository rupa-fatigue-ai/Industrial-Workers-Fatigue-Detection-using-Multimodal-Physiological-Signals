# Industrial Worker Fatigue Detection Using Multimodal Physiological and Environmental Signals using TAN and GAN model

This repository presents a **multimodal fatigue detection system** designed for industrial workers using physiological and environmental signals. The system leverages **signal processing, feature engineering, and machine learning/deep learning models** to estimate fatigue levels with high accuracy.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Pipeline Walkthrough](#pipeline-walkthrough)
7. [Module Reference](#module-reference)
8. [Models](#models)
9. [Outputs](#outputs)
10. [Configuration & CLI Flags](#configuration--cli-flags)
11. [Reproducibility](#reproducibility)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

The goal is **binary classification** of worker physiological state:

| Label | Meaning |
|-------|---------|
| 0     | Non-Fatigued (Relaxed) |
| 1     | Elevated Fatigue (Medium or High) |

The pipeline covers the full ML lifecycle:

```
Raw CSV files
    │
    ▼
Data loading & marker-based labelling   [data_loader.py]
    │
    ▼
Sliding-window feature extraction       [feature_extraction.py]
    │
    ▼
Train / test split + scaling            [preprocessing.py]
    │
    ├──► ML training (RF, SVM, LR)      [training.py]
    │
    ├──► cGAN data augmentation         [training.py]
    │
    └──► DL training (LSTM, TAN v1/v2)  [training.py]
              │
              ▼
        Evaluation, plots, tables       [results.py]
```

---

## Dataset

The dataset is made using multimodal sensor data collected using wearable systems, signals synchronized and timestamped. Fatigue scores annotated using validated MFI questionnaires. The Dataset can be download at: https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R?usp=sharing

---

## Repository Structure

```bash
worker-fatigue-detection/
│
├── data/                  # Dataset (or download instructions)
├── notebooks/            # Jupyter notebooks
│   └── WORKER_FATIGUE.ipynb
├──src/
│   └──worker_fatigue/
│       ├── main.py                  ← one-command pipeline runner
│       ├── data_loader.py           ← loading, labelling, imputation
│       ├── feature_extraction.py    ← sliding-window feature extraction
│       ├── preprocessing.py         ← scaling, sequences, class weights
│       ├── model.py                 ← all ML + DL architectures
│       ├── utils.py                 ← evaluation, plots, SHAP/LIME
│       ├── training.py              ← training loops for all models
│       ├── results.py               ← comparison tables + visualisations
│       └── outputs/                 ← auto-created on first run
│            ├── model_results.csv
│            ├── models/
│            │   ├── Random_Forest.pkl
│            │   ├── SVM.pkl
│            │   ├── Logistic_Regression.pkl
│            │   ├── baseline_lstm.h5
│            │   ├── tan_v1.h5
│            │   ├── tan_v2.h5
│            │   ├── cgan_lstm.h5
│            │   └── cgan_generator.h5
│            └── plots/
│                 ├── Random_Forest_dashboard.png
│                 ├── SVM_dashboard.png
│                 ├── Logistic_Regression_dashboard.png
│                 ├── LSTM_dashboard.png
│                 ├── TAN_v1_dashboard.png
│                 ├── TAN_v2_dashboard.png
│                 ├── cGAN_LSTM_dashboard.png
│                 ├── comparison_bar.png
│                 ├── comparison_radar_line.png
│                 └── metrics_heatmap.png
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```
---

## Installation

### 1. Clone / copy the scripts

```bash
git clone <repo-url>
cd worker_fatigue
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

### 3. Install dependencies

**Core (required for ML models):**
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

**Deep Learning (required for LSTM / TAN / cGAN models):**
```bash
pip install tensorflow>=2.10
```

**Explainability (optional but recommended):**
```bash
pip install shap lime
```

**Full one-liner:**
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn \
            tensorflow shap lime
```

> **GPU note:** For faster DL training install `tensorflow-gpu` or the CUDA-enabled TensorFlow wheel matching your CUDA version. The code auto-detects GPU / MPS / CPU via `tf.device`.

---

## Quick Start

### Full pipeline (ML + DL + cGAN + results)

```bash
python main.py \
  --dataset_csv dataset/workers_dataset \
  --marker_csv  dataset/marker_info.csv \
  --output_dir  outputs
```

### ML only (fast, ~2 minutes)

```bash
python main.py \
  --dataset_csv dataset/workers_dataset \
  --marker_csv  dataset/marker_info.csv \
  --output_dir  outputs \
  --skip_dl
```

### DL without cGAN augmentation

```bash
python main.py \
  --dataset_csv dataset/workers_dataset.csv \
  --marker_csv  dataset/marker_info.csv \
  --output_dir  outputs \
  --skip_cgan
```

### Re-use cached feature matrix (skip re-extraction)

```bash
python main.py \
  --dataset_csv dataset/workers_dataset.csv \
  --marker_csv  dataset/marker_info.csv \
  --output_dir  outputs \
  --feature_matrix_cache outputs/feature_matrix.csv
```

### Include SHAP + LIME explainability

```bash
python main.py \
  --dataset_csv dataset/workers_dataset.csv \
  --marker_csv  dataset/marker_info.csv \
  --output_dir  outputs \
  --explain
```

## Usage for Juypter Notebook
All the models and results have been compiled at one place and it can be used to view entire code with outputs at once. **Running jupyter notebook is preferred**.
```bash
jupyter notebook notebooks/WORKER_FATIGUE.ipynb
```

---

---

## Pipeline Walkthrough

### Step 1 — Data Loading (`data_loader.py`)

The Dataset can be download at: https://drive.google.com/drive/folders/133d4LgHQ6PHoftxWmM_0G7kPBhQPWX3R?usp=sharing. 
The worker dataset maps alphanumeric worker IDs to integers, derives:

- **ECG** 
- **EEG** 
- **GSR** 
- **TEMP** 
- **HR**
- **Spo2**
- **Environmental Noise level**
- **Environmental Dust density**

Then reads `marker_info.csv` to stamp each sample with one of three fatigue labels:

| Period | Label | Meaning |
|--------|-------|---------|
| NSP1, NSP2 | 0 | Non-Fatigued / Relaxed |
| MSP1, MSP2 | 1 | Medium Fatigue |
| HSP1, HSP2, HSP3 | 2 | High Fatigue |

Samples outside all defined periods are dropped. Missing values are imputed per worker via linear interpolation (zero-fill if 100 % missing).

### Step 2 — Feature Extraction (`feature_extraction.py`)

A **sliding-window** approach (200 s window, 10 s step) extracts **22 features** per window:

| Group | Features (8) |
|-------|-------------|
| ECG   | mean, var, std, rms, hrv_rmssd, hrv_sdnn, lf_hf_ratio, slope |
| GSR   | mean, var, std, slope, tonic_mean, phasic_std, scr_amp, peak_count |
| EEG   | mean, var, rms, energy, delta_power (1–4 Hz), theta_power (4–7.5 Hz) |

All frequency bands are constrained below the **7.75 Hz Nyquist limit**. HRV values are clipped at 1000 ms to remove physiologically impossible outliers.

Window labels are assigned by **majority vote** of per-sample labels within the window.

### Step 3 — Preprocessing (`preprocessing.py`)

- **Worker-level split**: workers 15 & 16 are held out as the test set (zero subject overlap / no data leakage)
- **Binary labels**: 3-class labels are collapsed to binary (0 = Relaxed, 1 = Elevated)
- **StandardScaler**: fit on training windows only, applied to both splits
- **Per-worker normalisation** + sequence building for DL models (`SEQ_LEN = 5`)
- **Class weights**: `{0: 1.8, 1: 1.0}` to handle imbalance

### Step 4 — Training (`training.py`)

#### ML Models

| Model | Threshold | Notes |
|-------|-----------|-------|
| Random Forest | 0.20 | 3 estimators, max_depth=5, balanced |
| SVM (RBF) | 0.20 | C=2.0, balanced |
| Logistic Regression | 0.20 | max_iter=1000, balanced |

#### cGAN Augmentation

A **Conditional GAN** is trained on the flat feature space to generate synthetic Non-Fatigued samples and balance the dataset. The Generator takes a noise vector + class label → outputs feature window. The Discriminator distinguishes real from synthetic windows.

#### DL Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| Baseline LSTM | LSTM(32) → LSTM(16) → Dense(1) | Dropout 0.4 |
| TAN v1 | LSTM(64) → LSTM(32) → SelfAttention → Dense(1) | Additive attention |
| TAN v2 (Full TAN) | LSTM(128) → LSTM(32) → [SelfAttn + GeneralAttn] → Dense(64) → Dense(1) | Dual attention |
| cGAN + LSTM | LSTM(64) → LSTM(32) → Dense(32) → Dense(1) | Trained on augmented data |

All DL models use EarlyStopping (patience=10) on `val_loss`.

### Step 5 — Results (`results.py`)

Generates:
- **Comparison table** sorted by test F1 (saved as CSV and printed)
- **Grouped bar chart** of test Accuracy / Precision / Recall / F1
- **Radar chart** + **Line chart** side-by-side
- **Metrics heatmap** across all models and metrics
- **(Optional)** SHAP aggregated feature importance + LIME instance explanation

---

## Module Reference

### `data_loader.py`

| Function | Description |
|----------|-------------|
| `load_compiled_csv(csv_path)` | Read and normalise the compiled CSV |
| `attach_labels(data, marker_csv)` | Stamp fatigue labels using marker timing |
| `impute_missing(data)` | Per-worker NaN imputation |
| `load_data(dataset_csv, marker_csv)` | Full loading pipeline |

### `feature_extraction.py`

| Function | Description |
|----------|-------------|
| `get_ecg_features(window)` | 8 ECG + HRV features |
| `get_gsr_features(window)` | 8 GSR tonic/phasic features |
| `get_eeg_features(window)` | 6 EEG band-power features |
| `extract_features(df)` | Full sliding-window extraction loop |
| `quality_check(fm)` | NaN / zero-value diagnostic |

### `preprocessing.py`

| Function | Description |
|----------|-------------|
| `worker_split(fm)` | No-leakage train/test split by worker |
| `binarise_labels(df)` | 3-class → binary |
| `scale_features(train, test)` | StandardScaler fit/transform |
| `normalise_per_worker(df)` | Per-worker Z-score for DL sequences |
| `build_sequences(df, scaler)` | (N, seq_len, n_features) arrays |
| `get_class_weights(y)` | Balanced class weights |
| `preprocess(fm)` | Full preprocessing pipeline |

### `model.py`

| Function | Description |
|----------|-------------|
| `build_random_forest()` | sklearn RandomForestClassifier |
| `build_svm()` | sklearn SVC (RBF) |
| `build_logistic_regression()` | sklearn LogisticRegression |
| `build_baseline_lstm(seq, feat)` | Keras stacked LSTM |
| `build_tan_v1(seq, feat)` | LSTM + SelfAttention |
| `build_tan_v2(seq, feat)` | LSTM + SelfAttention + GeneralAttention |
| `build_lstm_for_cgan(seq, feat)` | LSTM trained on augmented data |
| `build_cgan_generator(n_feat)` | cGAN Generator |
| `build_cgan_discriminator(n_feat)` | cGAN Discriminator |
| `build_cgan(gen, disc)` | Combined cGAN model |

### `utils.py`

| Function | Description |
|----------|-------------|
| `evaluate(y_true, y_prob, y_pred)` | ML model metrics + curves |
| `evaluate_dl(y_true, y_prob)` | DL metrics with auto-threshold tuning |
| `plot_dashboard(metrics, ...)` | 2×3 ML performance dashboard |
| `plot_dashboard_dl(metrics, ...)` | 2×4 DL performance dashboard |
| `model_results_append(...)` | Append to global results registry |
| `run_shap(model, ...)` | SHAP feature importance |
| `run_lime(model, ...)` | LIME instance explanation |

### `training.py`

| Function | Description |
|----------|-------------|
| `train_ml_models(prep, ...)` | Train RF, SVM, LR |
| `train_cgan(train_df, ...)` | Train cGAN + generate augmented data |
| `train_dl_models(prep, ...)` | Train LSTM, TAN v1, TAN v2, cGAN+LSTM |
| `train_all(prep, ...)` | Orchestrate full training |

### `results.py`

| Function | Description |
|----------|-------------|
| `build_comparison_table(results)` | Ranked comparison DataFrame |
| `filter_for_paper(df)` | Remove intermediate model entries |
| `plot_bar_comparison(df)` | Grouped bar chart |
| `plot_radar_line(df)` | Radar + line chart |
| `plot_metrics_heatmap(df)` | Full metrics heatmap |
| `generate_results(results, ...)` | Master result generation |

---

## Models

### Attention Mechanisms

**SelfAttention** (Bahdanau-style):
```
e_t = tanh(W · h_t + b)
a_t = softmax(e)
context = Σ a_t · h_t
```

**GeneralAttention** (Luong-style):
```
score = H · W · H^T
a     = softmax(score)
context = Σ a · H
```

TAN v2 concatenates both attention vectors before the dense head, giving the model both a soft global summary (self-attention) and a relational view across time steps (general attention).


#### Baseline Models for comparision

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Baseline LSTM
* LSTM + Attention mechanisms
* Conditional GAN (cGAN) for data augmentation

---

## Results & Performance Comparison

The following table summarizes model performance across key evaluation metrics:

| Model                                 | Accuracy   | Precision | Recall     | F1 Score   |
| ------------------------------------- | ---------- | --------- | ---------- | ---------- |
| **TAN (LSTM + Self Attention)**       | **0.9491** | **0.9373**| **0.9928** | **0.9603** |
| TAN (LSTM + Self + General Attention) | 0.9367     | 0.9217    | 0.9928     | 0.9409     |
| cGAN + LSTM                           | 0.9082     | 0.8992    | 0.9767     | 0.9363     |
| Baseline LSTM                         | 0.8859     | 0.8768    | 0.9713     | 0.9216     |
| Logistic Regression                   | 0.8775     | 0.8478    | 0.9900     | 0.9176     |
| SVM                                   | 0.8591     | 0.8289    | 0.9980     | 0.9064     |
| Random Forest                         | 0.8358     | 0.8591    | 0.9084     | 0.8831     |

---

---

## Outputs

After a full run, `outputs/` will contain:

| File | Description |
|------|-------------|
| `loaded_data.csv` | Labelled raw signal data |
| `feature_matrix.csv` | 22-feature windows with worker + label |
| `model_results.csv` | Raw per-model metric dict list |
| `model_comparison_full.csv` | Ranked comparison table (all models) |
| `model_comparison_paper.csv` | Filtered table for paper |
| `models/*.pkl` | Serialised sklearn models |
| `models/*.h5` | Saved Keras models |
| `plots/*_dashboard.png` | Per-model performance dashboards |
| `plots/comparison_bar.png` | Grouped bar comparison |
| `plots/comparison_radar_line.png` | Radar + line chart |
| `plots/metrics_heatmap.png` | Heatmap of all metrics |
| `plots/xai/shap_importance.png` | SHAP feature importance (if --explain) |
| `plots/xai/lime_explanation.png` | LIME instance explanation (if --explain) |

---

## Configuration & CLI Flags

All scripts expose CLI arguments. The most commonly used ones via `main.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_csv` | `dataset/workers_dataset.csv` | Compiled signal CSV |
| `--marker_csv` | `dataset/marker_info.csv` | Period timing CSV |
| `--output_dir` | `outputs` | Output directory |
| `--skip_dl` | False | Train only ML models |
| `--skip_cgan` | False | Skip GAN augmentation |
| `--explain` | False | Run SHAP + LIME |
| `--feature_matrix_cache` | None | Path to pre-built feature matrix |

Each sub-script can also be run independently:

```bash
# Data loading only
python data_loader.py \
  --dataset_csv dataset/workers_dataset.csv \
  --marker_csv  dataset/marker_info.csv \
  --output      outputs/loaded_data.csv

# Feature extraction only
python feature_extraction.py \
  --input  outputs/loaded_data.csv \
  --output outputs/feature_matrix.csv

# Preprocessing only
python preprocessing.py \
  --feature_matrix outputs/feature_matrix.csv \
  --output_dir     outputs

# Training only
python training.py \
  --feature_matrix outputs/feature_matrix.csv \
  --output_dir     outputs [--skip_dl] [--skip_cgan]

# Results from existing CSV
python results.py \
  --results_csv outputs/model_results.csv \
  --output_dir  outputs [--explain]
```

---

## Reproducibility

Global seeds are set in `training.py::set_seeds(42)`:

```python
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
```

The train/test split is **deterministic by worker ID** (workers 15 & 16 are always the test set), not by random sampling, so results are fully reproducible across runs without needing to fix any random state for the split itself.

> Note: GPU non-determinism can still cause small variation in DL results even with seeds fixed. For exact reproducibility on GPU, set `TF_DETERMINISTIC_OPS=1` and use `tensorflow-determinism`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
Install TensorFlow or use `--skip_dl` to run only ML models.

**`ModuleNotFoundError: No module named 'shap'` / `'lime'`**
Install with `pip install shap lime` or omit the `--explain` flag.

**`AssertionError: Missing column: ECG`**
The compiled CSV column names must match exactly. Check that `EMG` (not `EEG`) is the column name in `workers_dataset.csv`.

**`FileNotFoundError: marker_info.csv`**
Ensure `marker_info.csv` exists in the dataset directory and follows the format described above.

**`Worker leakage detected`**
This should never happen under normal usage. If you customise `TEST_WORKERS`, ensure the same worker does not appear in the training data.

**Slow feature extraction**
Feature extraction is the most time-consuming step (~3–10 min depending on hardware). Use `--feature_matrix_cache outputs/feature_matrix.csv` on subsequent runs to skip it.

**Out-of-memory on GPU during cGAN grid search**
Reduce `CGAN_BATCH` in `training.py` or use `--skip_cgan`.

---

##  Author
Dr. Sanchita Paul \
Dr. Rishabh Raj \
Birla Institute of Technology  

This work is part of a study on:

**Industrial Worker Fatigue Detection Using Multimodal Physiological and Environmental Signals**


---

##  Contributing

Contributions are welcome. Please fork the repo and submit a pull request.

---

##  License

MIT License
