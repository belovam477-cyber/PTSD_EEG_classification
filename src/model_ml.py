# model_ml.py
"""
Сравнение ML-моделей на наборе признаков B (без поведенческих).
Плюс пограничные поднаборы B1, B2 для попытки снизить FPR.

LOSO-валидация. Отбор признаков внутри фолда.

Best score -> B1 + SVM_RBF
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ===== КОНФИГ =====
DATA_PATH = Path("data/features_subject.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

K_FEATURES = 50
C_REG = 0.1
N_BOOTSTRAP = 2000

META_COLS = ["subject_id", "group", "age"]
# ==================


# ---- Наборы признаков ----
def is_not_behavior(name: str) -> bool:
    return not name.startswith("behavior_")


def is_not_technical(name: str) -> bool:
    """Убрать n_clean_segments."""
    return "n_clean_segments" not in name


def is_not_std_coh(name: str) -> bool:
    """Убрать schulte_std_coh_* (разброс coherence по пробам)."""
    return not (name.startswith("schulte_std_coh_"))


# ---- Модели ----
def get_models():
    return {
        "LR_L1": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=K_FEATURES)),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear",
                C=C_REG, max_iter=5000, class_weight="balanced",
            )),
        ]),
        "LR_L2": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=K_FEATURES)),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear",
                C=C_REG, max_iter=5000, class_weight="balanced",
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=K_FEATURES)),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True,
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=K_FEATURES)),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }


# ---- LOSO ----
def run_loso(pipe, X, y, groups):
    predictions = np.full(len(X), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        # Клонируем пайплайн через sklearn
        from sklearn.base import clone
        p = clone(pipe)
        p.fit(X[tr], y[tr])
        predictions[te] = p.predict_proba(X[te])[:, 1]
    return predictions


# ---- Метрики ----
def compute_metrics(df, predictions):
    pair = df["group"].isin(["ПТСР", "Норма"])
    y_pair = (df.loc[pair, "group"] == "ПТСР").astype(int).values
    p_pair = predictions[pair.values]

    auc = roc_auc_score(y_pair, p_pair)
    bal_acc = balanced_accuracy_score(y_pair, (p_pair > 0.5).astype(int))

    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_pair), len(y_pair))
        if len(np.unique(y_pair[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_pair[idx], p_pair[idx]))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])

    soma = df["group"] == "Соматоформные"
    p_soma = predictions[soma.values]
    fpr_soma = float((p_soma > 0.5).mean()) if len(p_soma) else float("nan")

    return {
        "auc": float(auc),
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "balanced_accuracy": float(bal_acc),
        "fpr_somatomorphic": fpr_soma,
    }


def evaluate(pipe, df, feature_cols, set_name, model_name):
    X = df[feature_cols].values
    y = (df["group"] == "ПТСР").astype(int).values
    groups = df["subject_id"].values

    predictions = run_loso(pipe, X, y, groups)
    metrics = compute_metrics(df, predictions)

    # Сохраняем предсказания и метрики
    df_out = df[["subject_id", "group", "age"]].copy()
    df_out["ptsd_prob"] = predictions
    df_out.to_csv(RESULTS_DIR / f"pred_{set_name}_{model_name}.csv", index=False)

    return metrics, predictions


def main():
    df = pd.read_csv(DATA_PATH)
    all_features = [c for c in df.columns if c not in META_COLS]

    # Три набора признаков
    features_B = [f for f in all_features if is_not_behavior(f)]
    features_B1 = [f for f in features_B if is_not_technical(f)]
    features_B2 = [f for f in features_B1 if is_not_std_coh(f)]

    feature_sets = {
        "B_no_behavior": features_B,
        "B1_no_tech": features_B1,
        "B2_no_std_coh": features_B2,
    }

    print(f"Наборы признаков:")
    for name, cols in feature_sets.items():
        print(f"  {name:20s} {len(cols)} признаков")
    print()

    models = get_models()
    print(f"Модели: {list(models.keys())}")
    print()

    # Сводная таблица результатов
    rows = []

    for set_name, cols in feature_sets.items():
        for model_name, pipe in models.items():
            print(f"[{set_name}] {model_name} ... ", end="", flush=True)
            metrics, preds = evaluate(pipe, df, cols, set_name, model_name)
            rows.append({
                "feature_set": set_name,
                "model": model_name,
                "n_features": len(cols),
                **metrics,
            })
            print(f"AUC={metrics['auc']:.3f}  FPR={metrics['fpr_somatomorphic']:.3f}")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("auc", ascending=False)

    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА (отсортировано по ROC-AUC)")
    print("=" * 80)
    print(summary.to_string(index=False))

    summary.to_csv(RESULTS_DIR / "ml_model_comparison.csv", index=False)
    with open(RESULTS_DIR / "ml_model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nСохранено:")
    print(f"  {RESULTS_DIR / 'ml_model_comparison.csv'}")
    print(f"  {RESULTS_DIR / 'ml_model_comparison.json'}")
    print(f"  pred_*.csv — предсказания по каждой модели")


if __name__ == "__main__":
    main()