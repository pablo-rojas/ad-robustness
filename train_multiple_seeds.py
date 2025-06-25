from train import main
import argparse
from src.misc_utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Uninformed Students with multiple seeds.")
    parser.add_argument('--type', type=str, default='us', choices=['us', 'mahalanobis', 'lid'], help='Type of detector to use.')
    parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
    parser.add_argument('--num_seeds', type=int, default=10, help='Number of random seeds.')
    parser.add_argument('--start', type=int, default=0, help='Starting seed index.')
    args = parser.parse_args()
    config = load_config(args.config)
    type = args.type

    # for loop over seeds
    for seed in range(args.start, args.num_seeds-1):
        seeded_config = config.copy()

        seeded_config['seed'] = seed
        seeded_config['experiment_name'] = f"{config['experiment_name']}_{seed}"
        seeded_config['model_path'] = f"{config['dataset']}_{type}_{seed}"
        
        # Call the main function with the updated config
        print(f"Training {type} with seed: {seed}")

        if type == 'us':
            from train import main
            main(seeded_config)
        else:
            from train_mahalanobis import main
            main(seeded_config, type=type, read_from_file=False)