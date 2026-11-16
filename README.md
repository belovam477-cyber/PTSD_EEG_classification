# EEG-based Classification of PTSD, Somatoform Disorders, and Healthy Controls

An end-to-end Machine Learning pipeline designed to differentiate between PTSD, Somatoform disorders, and Healthy controls using EEG (Electroencephalogram) data. 

## 🧠 Project Overview
This project leverages neurophysiological markers extracted from resting-state and cognitive task (Schulte tables) EEG recordings. The goal is to build a robust, interpretable model that can aid in clinical diagnostics and provide insights into the neurobiology of PTSD.

**Key accomplishments:**
*   Extracted advanced EEG features (aperiodic slope, alpha reactivity, hemispheric asymmetry, functional connectivity).
*   Implemented strict **Leave-One-Subject-Out (LOSO)** cross-validation to prevent data leakage and ensure generalization.
*   Achieved a **ROC-AUC of 0.949** and **Sensitivity of 0.84** for PTSD detection using an ensemble model (SVM + Logistic Regression).

## ⚙️ Key Features Extracted
Based on current neuroscience literature on PTSD, the pipeline extracts and analyzes:
1.  **Spectral Power:** Absolute, relative, and log power in Delta, Theta, Alpha, Beta, and Gamma bands.
2.  **Aperiodic Component:** Log-log slope and offset (reflecting cortical excitation/inhibition balance).
3.  **Hemispheric Asymmetry:** Log(left/right) power ratios (Frontal, Temporal, Occipital).
4.  **Functional Connectivity:** EEG coherence between 11 channel pairs.
5.  **Task-Rest Contrasts:** Alpha desynchronization during cognitive load.

## 📊 Results (LOSO Validation)
*All metrics are based on Leave-One-Subject-Out cross-validation to ensure subject-independent generalization.*

| Metric | Value |
|---|---|
| **ROC-AUC (PTSD vs Control)** | **0.949** [95% CI: 0.899, 0.986] |
| **Sensitivity (PTSD)** | **0.84** |
| **Specificity (Control)** | **0.94** |
| **False Positive Rate (Somatoform)** | **0.097** |

*Note: The final prediction is an ensemble of Logistic Regression (weight 0.65) and SVM with RBF kernel (weight 0.35).*

## 🛠️ Tech Stack
*   **Languages:** Python 3.13
*   **Signal Processing:** MNE-Python, SciPy
*   **Machine Learning:** Scikit-learn (SVM, Logistic Regression, SelectKBest)
*   **Data Analysis:** Pandas, NumPy

## 🚀 Repository Structure
*   `notebooks/` — Exploratory Data Analysis (EDA), feature visualization, and model training.
*   `src/` — Core pipeline scripts (`preprocess.py`, `features.py`, `evaluate.py`).
*   `submission/` — Standalone script for generating final predictions on unseen data.
*   `results/` — Model comparison metrics, ROC curves, and feature importance plots.

## 📁 How to Use
*To run the evaluation or make predictions, please refer to the technical instructions in the `submission/README.md`.*

## ⚠️ Data Privacy
The raw EEG data (`.edf` files) and patient metadata are **not** included in this repository due to medical privacy restrictions. The code is provided for pipeline demonstration and reproducibility purposes.