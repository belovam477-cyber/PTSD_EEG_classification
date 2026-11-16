import pandas as pd
from sklearn.metrics import confusion_matrix

meta = pd.read_csv("data/metadata.csv")

models = [
    ("LR_L1", "B_no_behavior"),
    ("LR_L2", "B_no_behavior"),
    ("RandomForest", "B_no_behavior"),
    ("SVM_RBF", "B1_no_tech"),
    ("SVM_RBF", "B_no_behavior"),
    ("RandomForest", "B1_no_tech"),
]

print(f"{'model':20s} {'feature_set':20s} {'AUC':>6s} {'Sens':>6s} {'Spec':>6s} {'FPR_Soma':>10s}")
print("-" * 80)

for model, fset in models:
    path = f"results/pred_{fset}_{model}.csv"
    try:
        preds = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"  нет файла: {path}")
        continue

    # Подтягиваем group из metadata
    if "group" not in preds.columns:
        preds = preds.merge(meta[["subject_id", "group"]], on="subject_id", how="left")

    # Пара ПТСР/Норма
    pair = preds[preds["group"].isin(["ПТСР", "Норма"])]
    y_true = (pair["group"] == "ПТСР").astype(int)
    y_pred = (pair["ptsd_prob"] > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # FPR на Соматоформных
    soma = preds[preds["group"] == "Соматоформные"]
    fpr_soma = (soma["ptsd_prob"] > 0.5).mean() if len(soma) else float("nan")

    print(f"{model:20s} {fset:20s} {sens:6.2f} {spec:6.2f} {fpr_soma:10.3f}")