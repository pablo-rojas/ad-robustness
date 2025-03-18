from train import main
import argparse
from src.misc_utils import load_config
import numpy as np

def random_search(template_config, lr_range, batch_size_range, beta1_range, n_iter):
    for i in range(n_iter):
        np.random.seed()
        config = template_config.copy()
        lr = np.exp(np.random.uniform(np.log(lr_range[0]), np.log(lr_range[1])))
        lr = float(lr)
        lr = float(f"{lr:.1e}")
        batch_size = int(np.exp(np.random.uniform(np.log(batch_size_range[0]), np.log(batch_size_range[1]))))
        beta1 = np.exp(np.random.uniform(np.log(beta1_range[0]), np.log(beta1_range[1])))
        beta1 = float(beta1)
        beta1 = float(f"{beta1:.2e}")

        config['train']['learning_rate'] = lr
        config['train']['batch_size'] = batch_size
        config['train']['beta1'] = beta1
        config["experiment_name"] =  config["dataset"] + "_rs_lr_{}_bs_{}_beta1_{}".format(lr, batch_size, beta1)
        config["model_path"] = "models/" + config["dataset"] + "_rs_lr_{}_bs_{}_beta1_{}".format(lr, batch_size, beta1)
        print(config)
        main(config)

def increasing_grid_search(template_config, lr_range, batch_size_range, grid_sizes):
    """
    Performs a grid search where each stage uses a grid with more points.
    A set of already evaluated parameter combinations is maintained so that
    repeated evaluations are skipped.
    
    Parameters:
        template_config (dict): The base configuration to update.
        lr_range (list): [min_lr, max_lr] for the learning rate.
        batch_size_range (list): [min_bs, max_bs] for the batch size.
        grid_sizes (list): A list of integers. For each value 'g' a grid of g points
                           per parameter is created (in logarithmic space).
    """
    tested = set()
    
    for grid_size in grid_sizes:
        # Create a logarithmically spaced grid for both learning rate and batch size.
        lrs = np.exp(np.linspace(np.log(lr_range[0]), np.log(lr_range[1]), grid_size))
        batch_sizes = np.exp(np.linspace(np.log(batch_size_range[0]), np.log(batch_size_range[1]), grid_size))
        
        for lr in lrs:
            for bs in batch_sizes:
                bs = int(bs)  # Ensure batch size is an integer
                lr = float(lr)
                rounded_lr = float(f"{lr:.1e}")
                # Skip if this combination was already evaluated.
                if (rounded_lr, bs) in tested:
                    continue

                config = template_config.copy()
                config['train']['learning_rate'] = rounded_lr
                config['train']['batch_size'] = bs
                config["experiment_name"] = f"{config['dataset']}_grid_lr_{lr}_bs_{bs}"
                config["model_path"] = f"models/{config['dataset']}_grid_lr_{lr}_bs_{bs}"
                main(config)

                tested.add((rounded_lr, bs))

parser = argparse.ArgumentParser(description="Run a random search for the CIFAR-10 dataset.")
parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
args = parser.parse_args()

template_config = load_config(args.config)

lr_range = [1e-5, 1e-2]
batch_size_range = [1, 128]
beta1_range = [0.7, 0.999]
n_iter = 1000

#increasing_grid_search(template_config, lr_range, batch_size_range, [2, 3, 4, 5, 6])

random_search(template_config, lr_range, batch_size_range, beta1_range, n_iter)