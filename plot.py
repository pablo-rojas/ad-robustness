import os
import json
import argparse
from eval_utils import save_histogram, save_cdf, save_roc_curve, save_inv_cdf
import numpy as np

from eval_utils import save_histogram, save_cdf, save_roc_curve, save_inv_cdf

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

range=(-3, 17)
bins=100

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate the detector model.")
parser.add_argument('--config', type=str, default='cfg/config.json', help='Path to the configuration file.')
args = parser.parse_args()

# Load configuration from JSON
config = load_config(args.config)

# General configuration
results_folder = "results/" + config['experiment_name']

# Load the results dictionary
results = np.load(os.path.join(results_folder, "results.npy"), allow_pickle=True).item()

# Plot and save histogram
save_histogram(results_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)

# Plot and save CDF
save_cdf(results_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)
save_inv_cdf(results_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)


# Plot and save ROC curve
save_roc_curve(results_folder, results['nat_list'], results['adv_list'])