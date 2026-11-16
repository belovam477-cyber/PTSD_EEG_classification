"""
Submission для задачи нейрофизиологической диагностики ПТСР.
predict(model, subject_dir) -> float
run(model, input_dir, output_csv) -> None
"""
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import mne
import pandas as pd
from scipy.signal import welch, coherence

warnings.filterwarnings("ignore")

# ===== КОНФИГ (должен совпадать с features.py) =====
EXPECTED_CHANNELS = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]
TARGET_FS = 125.0
BANDPASS = (0.5, 45.0)
NOTCH_FREQ = 50.0
REST_MAX_SEC = 60.0
MIN_SCHULTE_SEC = 3.0
ARTIFACT_MAD_THRESHOLD = 5.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
WELCH_NPERSEG = 250
WELCH_NOVERLAP = 125
MAX_BAD_FRACTION_IN_SEGMENT = 0.3

CONNECTIVITY_PAIRS = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 4), (2, 3), (2, 5), (3, 5), (4, 5),
]
ASYMMETRY_PAIRS = {
    "Fp1_Fp2": (2, 3),
    "T3_T4":   (1, 4),
    "O1_O2":   (0, 5),
}
# ===================================================


# ---------- Предобработка (копия preprocess.py) ----------
def parse_trial_num(filename: str) -> int:
    stem = Path(filename).stem
    if "П" in stem:
        return 0
    import re
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else -1


def read_and_preprocess_edf(path: Path, trial_type: str) -> Optional[np.ndarray]:
    """Читает и предобрабатывает один EDF. Возвращает (6, N) или None."""
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    except Exception:
        return None

    rename = {ch: ch.replace("EEG ", "").strip() for ch in raw.ch_names}
    raw.rename_channels(rename)

    missing = [ch for ch in EXPECTED_CHANNELS if ch not in raw.ch_names]
    if missing:
        return None

    raw.pick(EXPECTED_CHANNELS)

    if abs(raw.info["sfreq"] - TARGET_FS) > 0.01:
        raw.resample(TARGET_FS, verbose=False)

    raw.filter(BANDPASS[0], BANDPASS[1], verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    if trial_type == "rest":
        duration = raw.n_times / raw.info["sfreq"]
        if duration > REST_MAX_SEC:
            raw.crop(tmin=0, tmax=REST_MAX_SEC, verbose=False)

    duration = raw.n_times / raw.info["sfreq"]
    if trial_type == "schulte" and duration < MIN_SCHULTE_SEC:
        return None

    data = raw.get_data()
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True) * 1.4826
    mad[mad == 0] = 1e-6
    data_norm = (data - med) / mad

    return data_norm.astype(np.float32)


# ---------- Признаки (копия features.py) ----------
def bandpower_from_psd(freqs, psd, band):
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() == 0:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def clean_welch(data, fs=TARGET_FS):
    n_ch, n = data.shape
    step = WELCH_NPERSEG - WELCH_NOVERLAP
    bad = np.abs(data) > ARTIFACT_MAD_THRESHOLD
    psds = []
    for start in range(0, n - WELCH_NPERSEG + 1, step):
        end = start + WELCH_NPERSEG
        bad_frac = bad[:, start:end].any(axis=0).mean()
        if bad_frac > MAX_BAD_FRACTION_IN_SEGMENT:
            continue
        f, pxx = welch(data[:, start:end], fs=fs,
                       nperseg=WELCH_NPERSEG, noverlap=0, axis=-1)
        psds.append(pxx)
    if len(psds) == 0:
        f, pxx = welch(data, fs=fs, nperseg=WELCH_NPERSEG,
                       noverlap=WELCH_NOVERLAP, axis=-1)
        return f, pxx, 0
    return f, np.mean(psds, axis=0), len(psds)


def extract_spectral_features(data, prefix=""):
    f, psd, n_seg = clean_welch(data)
    n_ch = data.shape[0]
    feats = {}
    total_mask = (f >= 1.0) & (f <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], f[total_mask], axis=-1)
    total_power[total_power == 0] = 1e-12

    band_powers = {}
    for band_name, band_range in BANDS.items():
        bp = np.array([bandpower_from_psd(f, psd[ch], band_range)
                       for ch in range(n_ch)])
        band_powers[band_name] = bp
        for ch_idx, ch_name in enumerate(EXPECTED_CHANNELS):
            feats[f"{prefix}power_{band_name}_{ch_name}"] = bp[ch_idx]
            feats[f"{prefix}rel_{band_name}_{ch_name}"] = bp[ch_idx] / total_power[ch_idx]
            feats[f"{prefix}log_{band_name}_{ch_name}"] = np.log10(bp[ch_idx] + 1e-12)

    for ch_idx, ch_name in enumerate(EXPECTED_CHANNELS):
        theta = band_powers["theta"][ch_idx] + 1e-12
        alpha = band_powers["alpha"][ch_idx] + 1e-12
        beta = band_powers["beta"][ch_idx] + 1e-12
        feats[f"{prefix}theta_beta_{ch_name}"] = theta / beta
        feats[f"{prefix}alpha_theta_{ch_name}"] = alpha / theta

    alpha_mask = (f >= 8) & (f <= 13)
    if alpha_mask.sum() > 0:
        for ch_idx, ch_name in enumerate(EXPECTED_CHANNELS):
            alpha_psd = psd[ch_idx, alpha_mask]
            peak_idx = np.argmax(alpha_psd)
            feats[f"{prefix}alpha_peak_freq_{ch_name}"] = f[alpha_mask][peak_idx]
            feats[f"{prefix}alpha_peak_power_{ch_name}"] = alpha_psd[peak_idx]

    fit_mask = (f >= 2) & (f <= 40)
    if fit_mask.sum() > 3:
        log_f = np.log10(f[fit_mask])
        for ch_idx, ch_name in enumerate(EXPECTED_CHANNELS):
            log_p = np.log10(psd[ch_idx, fit_mask] + 1e-12)
            A = np.vstack([log_f, np.ones_like(log_f)]).T
            coef, *_ = np.linalg.lstsq(A, log_p, rcond=None)
            feats[f"{prefix}aperiodic_slope_{ch_name}"] = coef[0]
            feats[f"{prefix}aperiodic_offset_{ch_name}"] = coef[1]

    feats[f"{prefix}n_clean_segments"] = n_seg
    return feats


def extract_asymmetry_features(psd_feats, prefix=""):
    feats = {}
    for pair_name, (left_idx, right_idx) in ASYMMETRY_PAIRS.items():
        left_ch = EXPECTED_CHANNELS[left_idx]
        right_ch = EXPECTED_CHANNELS[right_idx]
        for band_name in BANDS:
            left = psd_feats.get(f"{prefix}power_{band_name}_{left_ch}", 1e-12) + 1e-12
            right = psd_feats.get(f"{prefix}power_{band_name}_{right_ch}", 1e-12) + 1e-12
            feats[f"{prefix}asym_{band_name}_{pair_name}"] = np.log10(left / right)
    return feats


def extract_connectivity_features(data, prefix=""):
    feats = {}
    for (i, j) in CONNECTIVITY_PAIRS:
        f, Cxy = coherence(data[i], data[j], fs=TARGET_FS,
                           nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
        for band_name in ["theta", "alpha", "beta"]:
            lo, hi = BANDS[band_name]
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                feats[f"{prefix}coh_{band_name}_{EXPECTED_CHANNELS[i]}_{EXPECTED_CHANNELS[j]}"] = float(Cxy[mask].mean())
    return feats


def extract_record_features(data, record_type):
    prefix = f"{record_type}_"
    feats = {}
    spec = extract_spectral_features(data, prefix=prefix)
    feats.update(spec)
    feats.update(extract_asymmetry_features(spec, prefix=prefix))
    feats.update(extract_connectivity_features(data, prefix=prefix))
    return feats


def aggregate_schulte_trials(schulte_feats):
    if not schulte_feats:
        return {}
    trials = sorted(schulte_feats.keys())
    prefix_len = len(f"trial{trials[0]}_")
    base_keys = [k[prefix_len:] for k in schulte_feats[trials[0]].keys()]
    out = {}
    for base_key in base_keys:
        vals, trial_nums = [], []
        for t in trials:
            v = schulte_feats[t].get(f"trial{t}_{base_key}")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
                trial_nums.append(t)
        if not vals:
            continue
        vals_arr = np.array(vals, dtype=float)
        out[f"schulte_mean_{base_key}"] = float(np.mean(vals_arr))
        if len(vals_arr) >= 2:
            out[f"schulte_std_{base_key}"] = float(np.std(vals_arr, ddof=0))
        else:
            out[f"schulte_std_{base_key}"] = np.nan
        out[f"schulte_first_{base_key}"] = float(vals_arr[0])
        if len(vals_arr) >= 2:
            out[f"schulte_last_{base_key}"] = float(vals_arr[-1])
        if len(vals_arr) >= 3:
            x = np.array(trial_nums, dtype=float)
            x_c = x - x.mean()
            out[f"schulte_slope_{base_key}"] = float(np.sum(x_c * (vals_arr - vals_arr.mean())) / np.sum(x_c ** 2))
        else:
            out[f"schulte_slope_{base_key}"] = np.nan
    return out


def extract_features_from_subject_dir(subject_dir: Path) -> Optional[dict]:
    """Извлекает все признаки для одного испытуемого."""
    edf_files = sorted(subject_dir.glob("*.edf"))
    if not edf_files:
        return None

    feats = {}
    rest_data = None
    schulte_data = {}
    schulte_durations = []

    for f in edf_files:
        trial_num = parse_trial_num(f.name)
        trial_type = "rest" if trial_num == 0 else "schulte"

        data = read_and_preprocess_edf(f, trial_type)
        if data is None:
            continue

        if trial_num == 0:
            rest_data = data
        else:
            schulte_data[trial_num] = data
            schulte_durations.append((trial_num, data.shape[1] / TARGET_FS))

    if rest_data is not None:
        feats.update(extract_record_features(rest_data, "rest"))
    else:
        feats["rest_missing"] = True

    schulte_feats = {}
    for n, data in schulte_data.items():
        schulte_feats[n] = extract_record_features(data, f"trial{n}")

    if schulte_feats:
        feats.update(aggregate_schulte_trials(schulte_feats))

    if schulte_durations:
        durations = np.array([d for _, d in schulte_durations])
        feats["behavior_mean_duration"] = float(durations.mean())
        feats["behavior_std_duration"] = float(durations.std(ddof=0))
        feats["behavior_max_duration"] = float(durations.max())
        feats["behavior_min_duration"] = float(durations.min())
        feats["behavior_sum_duration"] = float(durations.sum())
        if len(durations) >= 3:
            x = np.array([n for n, _ in schulte_durations], dtype=float)
            x_c = x - x.mean()
            feats["behavior_slope_duration"] = float(np.sum(x_c * (durations - durations.mean())) / np.sum(x_c ** 2))
        feats["behavior_first_duration"] = float(durations[0])
        feats["behavior_last_duration"] = float(durations[-1])

    # Контрасты
    if rest_data is not None and schulte_feats:
        for band in BANDS:
            for ch in EXPECTED_CHANNELS:
                rest_val = feats.get(f"rest_log_{band}_{ch}")
                task_val = feats.get(f"schulte_mean_log_{band}_{ch}")
                if rest_val is not None and task_val is not None:
                    feats[f"contrast_log_{band}_{ch}"] = task_val - rest_val

    return feats


# ---------- Предсказание ----------
def predict(model, subject_dir: Path) -> float:
    """
    Предсказание вероятности ПТСР для одной папки испытуемого.
    model: dict с ключами 'pipe_lr', 'features_B', 'pipe_svm', 'features_B1', 'w_svm', 'w_lr'
    """
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)

    feats = extract_features_from_subject_dir(subject_dir)
    if feats is None:
        return 0.5

    # LR — на features_B
    X_lr = pd.DataFrame([{c: feats.get(c, np.nan) for c in model["features_B"]}])
    X_lr = X_lr.fillna(0)
    prob_lr = model["pipe_lr"].predict_proba(X_lr.values)[:, 1][0]

    # SVM — на features_B1
    X_svm = pd.DataFrame([{c: feats.get(c, np.nan) for c in model["features_B1"]}])
    X_svm = X_svm.fillna(0)
    prob_svm = model["pipe_svm"].predict_proba(X_svm.values)[:, 1][0]

    prob = model["w_lr"] * prob_lr + model["w_svm"] * prob_svm
    return float(np.clip(prob, 0.0, 1.0))


def run(model, input_dir: Path, output_csv: Path) -> None:
    """
    Прогон по всем папкам испытуемых в input_dir.
    Сохраняет CSV с колонками subject_id, ptsd_probability.
    """
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_csv, str):
        output_csv = Path(output_csv)

    # Ищем подпапки-испытуемые рекурсивно
    subject_dirs = [d for d in input_dir.rglob("*") if d.is_dir() and any(d.glob("*.edf"))]

    rows = []
    for subj_dir in subject_dirs:
        try:
            prob = predict(model, subj_dir)
            rows.append({
                "subject_id": subj_dir.name,
                "ptsd_probability": prob,
            })
        except Exception as e:
            rows.append({
                "subject_id": subj_dir.name,
                "ptsd_probability": float("nan"),
            })
            print(f"Ошибка на {subj_dir}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Сохранено: {output_csv} ({len(df)} строк)")


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Использование: python predict.py <input_dir> <output_csv>")
        print("Пример: python submission/predict.py data/raw/Норма results/output.csv")
        sys.exit(1)
    
    model_path = Path("model/final_model.pkl")
    if not model_path.exists():
        # Попробуем рядом со скриптом
        model_path = Path(__file__).parent / "final_model.pkl"
    
    model = load_model(model_path)
    print(f"Модель загружена из {model_path}")
    run(model, Path(sys.argv[1]), Path(sys.argv[2]))