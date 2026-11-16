"""Уточнение веса ансамбля SVM + LR. Только усреднение, без обучения."""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

lr = pd.read_csv("results/pred_B_no_behavior_LR_L1.csv", encoding="utf-8-sig")
lr = lr.rename(columns={"ptsd_prob": "prob_lr"})

svm = pd.read_csv("results/pred_svm_best_sens.csv", encoding="utf-8-sig")
svm = svm.rename(columns={"ptsd_prob": "prob_svm"})

merged = lr.merge(svm[["subject_id", "prob_svm"]], on="subject_id")

print(f"{'w_svm':>6s} {'AUC':>6s} {'Sens':>6s} {'Spec':>6s} {'FPR':>6s}")
print("-" * 40)

for w_svm in [0.30, 0.35, 0.40, 0.45, 0.50]:
    merged["prob"] = w_svm * merged["prob_svm"] + (1 - w_svm) * merged["prob_lr"]

    pair = merged[merged["group"].isin(["ПТСР", "Норма"])]
    y = (pair["group"] == "ПТСР").astype(int)
    y_pred = (pair["prob"] > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    auc = roc_auc_score(y, pair["prob"])

    soma = merged[merged["group"] == "Соматоформные"]
    fpr = (soma["prob"] > 0.5).mean()

    print(f"{w_svm:>6.2f} {auc:>6.3f} {sens:>6.2f} {spec:>6.2f} {fpr:>6.3f}")