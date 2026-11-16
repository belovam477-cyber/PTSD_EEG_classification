"""
Предобработка ЭЭГ: чтение, ресемплинг, фильтрация, нормализация,
отбраковка артефактов. Сохраняет очищенные записи в data/clean/.
"""
import warnings
from pathlib import Path
import re

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ========== КОНФИГ ==========
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/clean")
META_PATH = Path("data/metadata.csv")
QC_OUT_PATH = Path("data/edf_qc_full.csv")

EXPECTED_CHANNELS = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]
TARGET_FS = 125.0          # единая частота дискретизации
BANDPASS = (0.5, 45.0)     # полоса пропускания
NOTCH_FREQ = 50.0          # сетевая помеха
REST_MAX_SEC = 60.0        # максимальная длина покоя (обрезаем сверху)
MIN_SCHULTE_SEC = 3.0      # минимальная длина Шульте
ARTIFACT_MAD_THRESHOLD = 5.0   # порог артефактов в единицах MAD
# ============================


def parse_trial_num(filename: str) -> int:
    """T-1..T-5 → 1..5, T-П → 0 (покой)."""
    stem = Path(filename).stem
    if "П" in stem:
        return 0
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else -1


def read_edf(path: Path) -> mne.io.BaseRaw:
    """Читает EDF и приводит каналы к ожидаемому набору."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    # Убираем возможные префиксы вроде "EEG "
    rename = {ch: ch.replace("EEG ", "").strip() for ch in raw.ch_names}
    raw.rename_channels(rename)

    missing = [ch for ch in EXPECTED_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels {missing}, got {raw.ch_names}")

    raw.pick_channels(EXPECTED_CHANNELS, ordered=True)
    return raw


def preprocess_one_edf(path: Path, trial_type: str) -> dict:
    """
    Читает и обрабатывает одну EDF-запись.
    trial_type: "rest" или "schulte"
    Возвращает dict с data (6, N), fs, bad_fraction или {"skipped": True}
    """
    raw = read_edf(path)

    # --- Ресемплинг к единой fs
    if abs(raw.info["sfreq"] - TARGET_FS) > 0.01:
        raw.resample(TARGET_FS, verbose=False)

    # --- Фильтрация
    raw.filter(BANDPASS[0], BANDPASS[1], verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    # --- Обрезка покоя
    if trial_type == "rest":
        duration = raw.n_times / raw.info["sfreq"]
        if duration > REST_MAX_SEC:
            raw.crop(tmin=0, tmax=REST_MAX_SEC, verbose=False)

    # --- Проверка длительности для Шульте
    duration = raw.n_times / raw.info["sfreq"]
    if trial_type == "schulte" and duration < MIN_SCHULTE_SEC:
        return {
            "skipped": True,
            "reason": f"schulte too short ({duration:.1f} s < {MIN_SCHULTE_SEC} s)",
            "duration": duration,
            "fs": raw.info["sfreq"],
        }

    # --- Нормализация MAD по каналу
    data = raw.get_data()  # (6, N)
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True) * 1.4826
    mad[mad == 0] = 1e-6
    data_norm = (data - med) / mad

    # --- Артефакты
    bad_mask = np.abs(data_norm) > ARTIFACT_MAD_THRESHOLD
    bad_fraction = float(bad_mask.any(axis=0).mean())

    return {
        "skipped": False,
        "data": data_norm.astype(np.float32),
        "fs": float(raw.info["sfreq"]),
        "duration": duration,
        "bad_fraction": bad_fraction,
        "n_samples": data_norm.shape[1],
    }


def process_subject(subject_dir: Path, subject_id: str, group: str) -> dict:
    """Обрабатывает одного испытуемого: все 6 EDF."""
    result = {"rest": None, "schulte": {}, "qc": []}

    for f in sorted(subject_dir.glob("*.edf")):
        trial_num = parse_trial_num(f.name)
        trial_type = "rest" if trial_num == 0 else "schulte"

        # Защита от пустых файлов
        if f.stat().st_size == 0:
            result["qc"].append({
                "subject_id": subject_id,
                "group": group,
                "file": f.name,
                "trial_num": trial_num,
                "trial_type": trial_type,
                "status": "empty",
                "duration": None,
                "fs": None,
                "bad_fraction": None,
                "n_samples": None,
                "reason": "file size = 0",
            })
            continue

        try:
            res = preprocess_one_edf(f, trial_type)
        except Exception as e:
            result["qc"].append({
                "subject_id": subject_id,
                "group": group,
                "file": f.name,
                "trial_num": trial_num,
                "trial_type": trial_type,
                "status": "error",
                "duration": None,
                "fs": None,
                "bad_fraction": None,
                "n_samples": None,
                "reason": str(e),
            })
            continue

        if res["skipped"]:
            result["qc"].append({
                "subject_id": subject_id,
                "group": group,
                "file": f.name,
                "trial_num": trial_num,
                "trial_type": trial_type,
                "status": "skipped",
                "duration": res["duration"],
                "fs": res["fs"],
                "bad_fraction": None,
                "n_samples": None,
                "reason": res["reason"],
            })
            continue

        if trial_type == "rest":
            result["rest"] = res["data"]
        else:
            result["schulte"][trial_num] = res["data"]

        result["qc"].append({
            "subject_id": subject_id,
            "group": group,
            "file": f.name,
            "trial_num": trial_num,
            "trial_type": trial_type,
            "status": "ok",
            "duration": round(res["duration"], 2),
            "fs": res["fs"],
            "bad_fraction": round(res["bad_fraction"], 4),
            "n_samples": res["n_samples"],
            "reason": None,
        })

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_PATH)
    meta = meta[meta["include"]].reset_index(drop=True)

    print(f"Испытуемых к обработке: {len(meta)}")
    print(f"Группы: {meta.groupby('group').size().to_dict()}\n")

    all_qc = []
    n_ok, n_partial, n_fail = 0, 0, 0

    for i, row in enumerate(meta.itertuples(), 1):
        subject_dir = Path(row.subject_dir)
        sid = row.subject_id
        group = row.group

        print(f"[{i:>3}/{len(meta)}] {group:20s} {sid}")

        res = process_subject(subject_dir, sid, group)
        all_qc.extend(res["qc"])

        # Считаем статус
        has_rest = res["rest"] is not None
        n_schulte = len(res["schulte"])

        if not has_rest and n_schulte == 0:
            n_fail += 1
            print(f"       → FAIL: нет валидных записей")
            continue
        elif not has_rest or n_schulte < 5:
            n_partial += 1
        else:
            n_ok += 1

        # Сохраняем
        subj_out = OUT_DIR / group / sid
        subj_out.mkdir(parents=True, exist_ok=True)

        if has_rest:
            np.save(subj_out / "rest.npy", res["rest"])
        for num, data in res["schulte"].items():
            np.save(subj_out / f"schulte_{num}.npy", data)

    # Сохраняем QC
    qc_df = pd.DataFrame(all_qc)
    qc_df.to_csv(QC_OUT_PATH, index=False)

    # Итог
    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print(f"Полных (6 записей):   {n_ok}")
    print(f"Частичных:            {n_partial}")
    print(f"Неудачных:            {n_fail}")
    print()

    if not qc_df.empty:
        print("Статусы файлов:")
        print(qc_df["status"].value_counts().to_dict())
        print()

        ok = qc_df[qc_df["status"] == "ok"]
        if len(ok) > 0:
            print("Средний bad_fraction по группам:")
            print(ok.groupby("group")["bad_fraction"].mean().round(4).to_dict())
            print()

            print("Длительность покоя после обрезки:")
            rest = ok[ok["trial_type"] == "rest"]
            if len(rest) > 0:
                print(rest["duration"].describe().round(2).to_dict())
            print()

            print("Длительность Шульте:")
            sch = ok[ok["trial_type"] == "schulte"]
            if len(sch) > 0:
                print(sch["duration"].describe().round(2).to_dict())

    print(f"\nQC сохранён в: {QC_OUT_PATH}")
    print(f"Очищенные данные: {OUT_DIR}")


if __name__ == "__main__":
    main()