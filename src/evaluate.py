import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import os


os.makedirs("results/figures", exist_ok=True)

meta = pd.read_csv("data/metadata.csv", encoding="utf-8-sig")

# Список моделей для оценки
models = [
    ("LR_L1", "B_no_behavior"),
    ("LR_L2", "B_no_behavior"),
    ("RandomForest", "B_no_behavior"),
    ("SVM_RBF", "B1_no_tech"),
    ("SVM_RBF", "B_no_behavior"),
    ("RandomForest", "B1_no_tech"),
]

# Функция для вычисления Baseline (мажоритарный класс)
print("=== BASELINE ===")
pair_meta = meta[meta["group"].isin(["ПТСР", "Норма"])]
majority_class = pair_meta["group"].value_counts().idxmax()
baseline_acc = pair_meta["group"].value_counts(normalize=True).max()
print(f"Majority class: {majority_class}, Baseline Accuracy: {baseline_acc:.3f}")
print("================\n")

print(f"{'model':20s} {'feature_set':20s} {'AUC':>6s} {'Sens':>6s} {'Spec':>6s} {'FPR_Soma':>10s}")
print("-" * 85)

best_auc = 0
best_model_name = ""

for model, fset in models:
    path_1 = f"results/pred_{fset}_{model}.csv"
    path_2 = f"results/_archive/pred_{fset}_{model}.csv"
    
    if os.path.exists(path_1):
        path = path_1
    elif os.path.exists(path_2):
        path = path_2
    else:
        print(f"  [!] нет файла: {path_1} (или в _archive)")
        continue

    try:
        preds = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [!] Ошибка чтения {path}: {e}")
        continue

    # Подтягиваем group из metadata
    if "group" not in preds.columns:
        preds = preds.merge(meta[["subject_id", "group"]], on="subject_id", how="left")

    # Пара ПТСР/Норма
    pair = preds[preds["group"].isin(["ПТСР", "Норма"])].copy()
    
    # Если вдруг пусто (несовпадение ID)
    if len(pair) == 0:
        print(f"  [!] Нет совпадений для {model} ({fset})")
        continue

    y_true = (pair["group"] == "ПТСР").astype(int)
    y_pred = (pair["ptsd_prob"] > 0.5).astype(int)
    
    # Считаем AUC
    auc = roc_auc_score(y_true, pair["ptsd_prob"])
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

    # FPR на Соматоформных
    soma = preds[preds["group"] == "Соматоформные"]
    fpr_soma = (soma["ptsd_prob"] > 0.5).mean() if len(soma) else float("nan")

    print(f"{model:20s} {fset:20s} {auc:6.3f} {sens:6.2f} {spec:6.2f} {fpr_soma:10.3f}")

    # Запоминаем лучшую модель для Confusion Matrix
    if auc > best_auc:
        best_auc = auc
        best_model_name = f"{model}_{fset}"
        best_cm = cm

# Сохраняем Confusion Matrix для лучшей модели
if best_model_name:
    print(f"\nСохранение Confusion Matrix для лучшей модели: {best_model_name}")
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["Control", "PTSD"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix: {best_model_name} (LOSO)")
    plt.savefig("results/figures/fig_confusion_matrix.png", dpi=120, bbox_inches='tight')
    plt.close() # Закрываем график, чтобы не блокировал выполнение
    print("Сохранено в results/figures/fig_confusion_matrix.png")