from src.eval_utils import save_results, partial_auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

results_dir = "./results/benchmark_cifar_ens_today/fgsm_inf_0.05"
results_dir = "./results/benchmark_cifar_today/pgd_inf_0.1"

results_us = np.load(results_dir + "/uninformed_students/results.npy", allow_pickle=True).item()
results_acgan = np.load(results_dir + "/acgan/results.npy",allow_pickle=True).item()


anomaly_free_scores = np.array(results_us['nat_list'])
anomalous_scores = np.array(results_us['adv_list'])
y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])

# Calculate the ROC curve
fpr_us, tpr_us, _ = roc_curve(y_true, y_scores)

anomaly_free_scores = np.array(results_acgan['nat_list'])
anomalous_scores = np.array(results_acgan['adv_list'])
y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])

# Calculate the ROC curve
fpr_acgan, tpr_acgan, _ = roc_curve(y_true, y_scores)

# Calculate the AUC
auc_score_us = auc(fpr_us, tpr_us)
auc_score_acgan = auc(fpr_acgan, tpr_acgan)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_us, tpr_us, color='blue', lw=2, label=f"US ROC curve (area = {auc_score_us:.2f})")
plt.plot(fpr_acgan, tpr_acgan, color='darkorange', lw=2, label=f"ACGAN ROC curve (area = {auc_score_acgan:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join("./", "roc_curve_pgd_0.1.png"))
plt.close()