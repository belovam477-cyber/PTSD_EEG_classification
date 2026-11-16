# features.py
"""
Извлечение признаков из очищенных ЭЭГ-записей.
Вход:  data/clean/<group>/<subject_id>/rest.npy, schulte_1..5.npy
Выход: data/features_subject.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence

warnings.filterwarnings("ignore")

# ===== КОНФИГ =====
CLEAN_DIR = Path("data/clean")
META_PATH = Path("data/metadata.csv")
OUT_PATH = Path("data/features_subject.csv")

FS = 125.0
CHANNELS = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]

# Частотные диапазоны (Гц)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Параметры Welch
WELCH_NPERSEG = 250       # 2 с
WELCH_NOVERLAP = 125      # 50%

# Порог отбраковки сегментов
MAD_THRESHOLD = 5.0
MAX_BAD_FRACTION_IN_SEGMENT = 0.3

# Пары каналов для связности (индексы в CHANNELS)
CONNECTIVITY_PAIRS = [
    (0, 1),  # O1-T3
    (0, 2),  # O1-Fp1
    (0, 3),  # O1-Fp2
    (0, 4),  # O1-T4
    (0, 5),  # O1-O2
    (1, 2),  # T3-Fp1
    (1, 4),  # T3-T4
    (2, 3),  # Fp1-Fp2
    (2, 5),  # Fp1-O2
    (3, 5),  # Fp2-O2
    (4, 5),  # T4-O2
]

# Пары для асимметрий (левое, правое)
ASYMMETRY_PAIRS = {
    "Fp1_Fp2": (2, 3),
    "T3_T4":   (1, 4),
    "O1_O2":   (0, 5),
}
# ==================


def bandpower_from_psd(freqs, psd, band):
    """Интеграл PSD по диапазону."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() == 0:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def clean_welch(data, fs=FS, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP):
    """
    Welch PSD, усреднённый только по чистым сегментам.
    data: (n_channels, n_samples), уже MAD-нормализованный.
    Возвращает (freqs, psd, n_clean_segments), где psd: (n_channels, n_freqs).
    """
    n_ch, n = data.shape
    step = nperseg - noverlap
    bad = np.abs(data) > MAD_THRESHOLD  # (n_ch, n)

    psds = []
    for start in range(0, n - nperseg + 1, step):
        end = start + nperseg
        bad_frac = bad[:, start:end].any(axis=0).mean()
        if bad_frac > MAX_BAD_FRACTION_IN_SEGMENT:
            continue
        f, pxx = welch(data[:, start:end], fs=fs,
                       nperseg=nperseg, noverlap=0,
                       axis=-1)
        psds.append(pxx)

    if len(psds) == 0:
        # Если чистых сегментов нет — считаем по всей записи
        f, pxx = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
        return f, pxx, 0

    psd_mean = np.mean(psds, axis=0)
    return f, psd_mean, len(psds)


def extract_spectral_features(data, fs=FS, prefix=""):
    """Спектральные признаки для одной записи."""
    f, psd, n_seg = clean_welch(data, fs)
    n_ch = data.shape[0]
    feats = {}

    # Общая мощность 1–45 Гц
    total_mask = (f >= 1.0) & (f <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], f[total_mask], axis=-1)
    total_power[total_power == 0] = 1e-12

    # Мощность по диапазонам
    band_powers = {}
    for band_name, band_range in BANDS.items():
        bp = np.array([
            bandpower_from_psd(f, psd[ch], band_range)
            for ch in range(n_ch)
        ])
        band_powers[band_name] = bp

        for ch_idx, ch_name in enumerate(CHANNELS):
            feats[f"{prefix}power_{band_name}_{ch_name}"] = bp[ch_idx]
            feats[f"{prefix}rel_{band_name}_{ch_name}"] = bp[ch_idx] / total_power[ch_idx]
            feats[f"{prefix}log_{band_name}_{ch_name}"] = np.log10(bp[ch_idx] + 1e-12)

    # Соотношения
    for ch_idx, ch_name in enumerate(CHANNELS):
        theta = band_powers["theta"][ch_idx] + 1e-12
        alpha = band_powers["alpha"][ch_idx] + 1e-12
        beta = band_powers["beta"][ch_idx] + 1e-12
        feats[f"{prefix}theta_beta_{ch_name}"] = theta / beta
        feats[f"{prefix}alpha_theta_{ch_name}"] = alpha / theta

    # Alpha peak
    alpha_mask = (f >= 8) & (f <= 13)
    if alpha_mask.sum() > 0:
        for ch_idx, ch_name in enumerate(CHANNELS):
            alpha_psd = psd[ch_idx, alpha_mask]
            peak_idx = np.argmax(alpha_psd)
            peak_freq = f[alpha_mask][peak_idx]
            feats[f"{prefix}alpha_peak_freq_{ch_name}"] = peak_freq
            feats[f"{prefix}alpha_peak_power_{ch_name}"] = alpha_psd[peak_idx]

    # Апериодический наклон (log-log фит 2–40 Гц)
    fit_mask = (f >= 2) & (f <= 40)
    if fit_mask.sum() > 3:
        log_f = np.log10(f[fit_mask])
        slopes = []
        offsets = []
        for ch_idx in range(n_ch):
            log_p = np.log10(psd[ch_idx, fit_mask] + 1e-12)
            A = np.vstack([log_f, np.ones_like(log_f)]).T
            coef, *_ = np.linalg.lstsq(A, log_p, rcond=None)
            slopes.append(coef[0])
            offsets.append(coef[1])
        for ch_idx, ch_name in enumerate(CHANNELS):
            feats[f"{prefix}aperiodic_slope_{ch_name}"] = slopes[ch_idx]
            feats[f"{prefix}aperiodic_offset_{ch_name}"] = offsets[ch_idx]

    feats[f"{prefix}n_clean_segments"] = n_seg
    return feats


def extract_asymmetry_features(psd_feats, prefix=""):
    """Асимметрии: log(left/right) по диапазонам."""
    feats = {}
    for pair_name, (left_idx, right_idx) in ASYMMETRY_PAIRS.items():
        left_ch = CHANNELS[left_idx]
        right_ch = CHANNELS[right_idx]
        for band_name in BANDS:
            left = psd_feats.get(f"{prefix}power_{band_name}_{left_ch}", 1e-12) + 1e-12
            right = psd_feats.get(f"{prefix}power_{band_name}_{right_ch}", 1e-12) + 1e-12
            feats[f"{prefix}asym_{band_name}_{pair_name}"] = np.log10(left / right)
    return feats


def extract_connectivity_features(data, fs=FS, prefix=""):
    """Coherence между парами каналов в theta/alpha/beta."""
    n_ch = data.shape[0]
    feats = {}
    for (i, j) in CONNECTIVITY_PAIRS:
        f, Cxy = coherence(data[i], data[j], fs=fs,
                           nperseg=WELCH_NPERSEG,
                           noverlap=WELCH_NOVERLAP)
        for band_name in ["theta", "alpha", "beta"]:
            lo, hi = BANDS[band_name]
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                feats[f"{prefix}coh_{band_name}_{CHANNELS[i]}_{CHANNELS[j]}"] = \
                    float(Cxy[mask].mean())
    return feats


def extract_record_features(data, record_type):
    """Все признаки для одной записи (покой или одна проба Шульте)."""
    prefix = f"{record_type}_"
    feats = {}
    spec = extract_spectral_features(data, prefix=prefix)
    feats.update(spec)
    asym = extract_asymmetry_features(spec, prefix=prefix)
    feats.update(asym)
    conn = extract_connectivity_features(data, prefix=prefix)
    feats.update(conn)
    return feats


def aggregate_schulte_trials(schulte_feats):
    """
    Из 5 наборов признаков Шульте делает агрегаты:
    mean, std, slope, first, last.
    Ключи без префикса trialN_.
    """
    if not schulte_feats:
        return {}

    trials = sorted(schulte_feats.keys())  # [1..5]

    # Убираем префикс trialN_ из ключей первого набора
    first_key = next(iter(schulte_feats[trials[0]].keys()))
    # first_key = "trial1_power_delta_O1"
    # Нужно получить "power_delta_O1"
    prefix_len = len(f"trial{trials[0]}_")
    base_keys = [k[prefix_len:] for k in schulte_feats[trials[0]].keys()]

    out = {}

    for base_key in base_keys:
        vals = []
        trial_nums = []
        for t in trials:
            full_key = f"trial{t}_{base_key}"
            v = schulte_feats[t].get(full_key)
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
            slope = float(np.sum(x_c * (vals_arr - vals_arr.mean())) / np.sum(x_c ** 2))
            out[f"schulte_slope_{base_key}"] = slope
        else:
            out[f"schulte_slope_{base_key}"] = np.nan

    return out


def extract_subject_features(subject_id, group, subject_dir):
    """Полный набор признаков для одного испытуемого."""
    feats = {"subject_id": subject_id, "group": group}

    # Покой
    rest_path = subject_dir / "rest.npy"
    if rest_path.exists():
        rest = np.load(rest_path)
        rest_feats = extract_record_features(rest, "rest")
        feats.update(rest_feats)
    else:
        feats["rest_missing"] = True

    # Шульте 1..5
    schulte_feats = {}
    schulte_durations = []
    for n in range(1, 6):
        p = subject_dir / f"schulte_{n}.npy"
        if not p.exists():
            continue
        data = np.load(p)
        dur = data.shape[1] / FS
        schulte_durations.append((n, dur))
        rec_feats = extract_record_features(data, f"trial{n}")
        schulte_feats[n] = rec_feats

    # Агрегация по 5 пробам
    if schulte_feats:
        agg = aggregate_schulte_trials(schulte_feats)
        feats.update(agg)

    # Поведенческие длительности
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
            slope = float(np.sum(x_c * (durations - durations.mean())) / np.sum(x_c ** 2))
            feats["behavior_slope_duration"] = slope
        feats["behavior_first_duration"] = float(durations[0])
        feats["behavior_last_duration"] = float(durations[-1])

    # Контрасты покой ↔ Шульте
    if rest_path.exists() and schulte_feats:
        for band in BANDS:
            for ch in CHANNELS:
                rest_val = feats.get(f"rest_log_{band}_{ch}")
                task_val = feats.get(f"schulte_mean_log_{band}_{ch}")
                if rest_val is not None and task_val is not None:
                    feats[f"contrast_log_{band}_{ch}"] = task_val - rest_val

    return feats


def main():
    meta = pd.read_csv(META_PATH)
    meta = meta[meta["include"]].reset_index(drop=True)
    print(f"Испытуемых: {len(meta)}")

    rows = []
    for i, row in enumerate(meta.itertuples(), 1):
        sid = row.subject_id
        group = row.group
        subj_dir = CLEAN_DIR / group / sid

        if not subj_dir.exists():
            print(f"[{i}] {sid}: папки нет в data/clean/ — пропуск")
            continue

        try:
            feats = extract_subject_features(sid, group, subj_dir)
            feats["age"] = row.age
            rows.append(feats)
            print(f"[{i:>3}/{len(meta)}] {group:20s} {sid}  "
                  f"({len(feats)} признаков)")
        except Exception as e:
            print(f"[{i:>3}/{len(meta)}] {sid}: ОШИБКА — {e}")

    df = pd.DataFrame(rows)

    # Удаляем дублирующие first/last
    drop_cols = [
    c for c in df.columns
    if c.startswith("schulte_first_") or c.startswith("schulte_last_")
]
    df = df.drop(columns=drop_cols)
    print(f"Удалено колонок: {len(drop_cols)}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nСохранено: {OUT_PATH}")
    print(f"Форма: {df.shape}")
    print(f"Колонки (первые 20): {list(df.columns[:20])}")


if __name__ == "__main__":
    main()