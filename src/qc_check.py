# qc_check.py
import mne
import pandas as pd
from pathlib import Path

EXPECTED = {"O1", "T3", "Fp1", "Fp2", "T4", "O2"}


def check_one(path: Path) -> dict:
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        fs = float(raw.info["sfreq"])
        duration = raw.n_times / fs
        ch_names = raw.ch_names
        ch_clean = set(ch.replace("EEG ", "").strip() for ch in ch_names)

        return {
            "fs": fs,
            "duration_sec": round(duration, 2),
            "n_channels": len(ch_names),
            "channels_raw": ";".join(ch_names),
            "channels_ok": ch_clean == EXPECTED,
            "error": None,
        }
    except Exception as e:
        return {
            "fs": None, "duration_sec": None, "n_channels": None,
            "channels_raw": None, "channels_ok": None,
            "error": str(e),
        }


def main():
    meta = pd.read_csv("data/metadata.csv")
    rows = []

    for _, r in meta.iterrows():
        subj_dir = Path(r["subject_dir"])
        for f in sorted(subj_dir.glob("*.edf")):
            info = check_one(f)
            rows.append({
                "subject_id": r["subject_id"],
                "group": r["group"],
                "file": f.name,
                **info,
            })
            print(f"{r['group']:20s} {r['subject_id']:12s} {f.name:12s} "
                  f"fs={info['fs']} dur={info['duration_sec']}")

    qc = pd.DataFrame(rows)
    qc.to_csv("data/edf_qc.csv", index=False)

    print("\n===== СВОДКА =====")
    print("Всего файлов:", len(qc))
    print("Ошибки чтения:", qc["error"].notna().sum())
    print("fs:", qc["fs"].value_counts().to_dict())
    print("Каналы OK:", qc["channels_ok"].sum(), "/", len(qc))
    print("\nДлительность покоя:")
    rest = qc[qc.file.str.contains("П")]
    print(rest["duration_sec"].describe())


if __name__ == "__main__":
    main()