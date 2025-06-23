from src.eval_utils import save_results, partial_auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def plot_roc_curves_for_methods(benchmark_dir, eval_dir, method_dirs):
    plt.figure(figsize=(10, 6))
    
    for method in method_dirs:
        results_path = os.path.join(benchmark_dir, eval_dir, method, "results.npy")
        if not os.path.exists(results_path):
            continue
            
        try:
            results = np.load(results_path, allow_pickle=True).item()
            
            anomaly_free_scores = np.array(results['nat_list'])
            anomalous_scores = np.array(results['adv_list'])
            y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
            y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])
            
            # Calculate the ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            
            # Calculate the AUC
            auc_score = auc(fpr, tpr)
            
            # Plot the ROC curve for this method
            plt.plot(fpr, tpr, lw=2, label=f"{method} ROC curve (area = {auc_score:.2f})")
            
        except Exception as e:
            print(f"Error processing {results_path}: {e}")
    
    # Add the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {benchmark_dir.split("/")[-1]} - {eval_dir}')
    plt.legend(loc="lower right")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(benchmark_dir, eval_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "ROC.png"))
    plt.close()

def main():
    results_root = "./results"
    
    # Iterate through all benchmarks
    for benchmark_name in os.listdir(results_root):
        benchmark_dir = os.path.join(results_root, benchmark_name)
        if not os.path.isdir(benchmark_dir):
            continue
            
        # Iterate through all evaluations for this benchmark
        for eval_name in os.listdir(benchmark_dir):
            eval_dir = os.path.join(benchmark_dir, eval_name)
            if not os.path.isdir(eval_dir):
                continue
                
            # Get all method directories for this evaluation (skip 'img' folder)
            method_dirs = [d for d in os.listdir(eval_dir) 
                          if os.path.isdir(os.path.join(eval_dir, d)) and d != 'img']
            
            if method_dirs:
                print(f"Processing {benchmark_name}/{eval_name} with methods: {', '.join(method_dirs)}")
                # Plot ROC curves for all methods in this evaluation
                plot_roc_curves_for_methods(benchmark_dir, eval_name, method_dirs)

if __name__ == "__main__":
    main()