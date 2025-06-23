from src.eval_utils import save_results, partial_auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def get_marker(eval_name):
    # For natural images
    if eval_name == "natural":
        return "o"
    if "fsgm" in eval_name:
        return "s"  # square
    elif "pgd_T_inf" in eval_name:
        return "^"  # triangle
    elif "pgd_T_2" in eval_name:
        return "D"  # diamond
    elif "cw" in eval_name:
        return "v"  # inverted triangle
    # Default marker if none of the keywords match
    return "x"

def plot_points_l2(benchmark_dir, eval_dirs, method_dirs):
    
    
    # Use a distinct color per eval_dir (shared across methods)
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_dirs)+1))
    
    for method in method_dirs:
        plt.figure(figsize=(10, 6))

        # Plot natural images first
        natural_path = os.path.join(benchmark_dir, eval_dirs[0], method, "results.npy")
        if os.path.exists(natural_path):
            results = np.load(natural_path, allow_pickle=True).item()
                
            anomaly_scores = np.array(results['nat_list'])
            l2_distances = np.zeros_like(anomaly_scores)
    
            plt.scatter(anomaly_scores, l2_distances, 
                        color=colors[0],
                        marker=get_marker("natural"),
                        label="Natural images")
        
        # Plot other evaluations
        for idx, eval_dir in enumerate(eval_dirs[1:], start=1):
            results_path = os.path.join(benchmark_dir, eval_dir, method, "results.npy")
            if not os.path.exists(results_path):
                continue
                
            results = np.load(results_path, allow_pickle=True).item()
            
            anomaly_scores = np.array(results['adv_list'])
            l2_distances = np.array(results['l2_dist'])
    
            plt.scatter(anomaly_scores, l2_distances, 
                        color=colors[idx],
                        marker=get_marker(eval_dir),
                        label=f"{eval_dir}")
        plt.xlabel("Anomaly Scores")
        plt.ylabel("L2 distances")
        plt.title(f"Scatter Plot {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_dir, "img", f"scatter_plot_l2_{method}.png"), dpi=300)
        plt.show()
        plt.close()

def plot_points_linf(benchmark_dir, eval_dirs, method_dirs):
    
    # Use a distinct color per eval_dir (shared across methods)
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_dirs)+1))
    
    for method in method_dirs:
        plt.figure(figsize=(10, 6))
        # Plot natural images first
        natural_path = os.path.join(benchmark_dir, eval_dirs[0], method, "results.npy")
        if os.path.exists(natural_path):
            results = np.load(natural_path, allow_pickle=True).item()
                
            anomaly_scores = np.array(results['nat_list'])
            linf_distances = np.zeros_like(anomaly_scores)
    
            plt.scatter(anomaly_scores, linf_distances, 
                        color=colors[0],
                        marker=get_marker("natural"),
                        label="Natural images")
        
        # Plot other evaluations
        for idx, eval_dir in enumerate(eval_dirs[1:], start=1):
            results_path = os.path.join(benchmark_dir, eval_dir, method, "results.npy")
            if not os.path.exists(results_path):
                continue
                
            results = np.load(results_path, allow_pickle=True).item()
            
            anomaly_scores = np.array(results['adv_list'])
            linf_distances = np.array(results['linf_dist'])
    
            plt.scatter(anomaly_scores, linf_distances, 
                        color=colors[idx],
                        marker=get_marker(eval_dir),
                        label=f"{eval_dir}")
        plt.xlabel("Anomaly Scores")
        plt.ylabel("Linf distances")
        plt.title(f"Scatter Plot {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_dir, "img", f"scatter_plot_linf_{method}.png"))
        plt.show()
        plt.close()
        
def main():
    benchmark_dir = "./results/cifar_plot"
    method_dirs = ['acgan', 'uninformed_students']
            
    # Get all evaluation directories for this benchmark (skip 'img' folder) and sort them alphabetically
    eval_dirs = sorted([d for d in os.listdir(benchmark_dir) 
                        if os.path.isdir(os.path.join(benchmark_dir, d)) and d != 'img'])
    # Ensure the first evaluation directory is for natural images
    # If not, you may want to manually insert "natural" or adjust accordingly.
    if method_dirs:
        plot_points_l2(benchmark_dir, eval_dirs, method_dirs)
        plot_points_linf(benchmark_dir, eval_dirs, method_dirs)

if __name__ == "__main__":
    main()