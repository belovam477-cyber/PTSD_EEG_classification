# Data Directory

This folder contains aggregated and anonymized tabular data extracted from the raw EEG recordings. 
Raw `.edf` files and personal health information are not included due to medical privacy restrictions.

## Files Description

*   **`metadata.csv`** — Subject IDs, age, group label (PTSD, Control, Somatoform).
*   **`features_subject.csv`** — Final extracted features (spectral power, aperiodic slope, connectivity) for each subject, used for model training.
*   **`edf_qc.csv` / `edf_qc_full.csv`** — Quality Control (QC) reports showing signal artifacts, bad segments, and data validity.
