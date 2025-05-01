from train import main
import argparse
from src.misc_utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Uninformed Students with multiple seeds.")
    parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
    parser.add_argument('--num_seeds', type=int, default=10, help='Number of random seeds.')
    args = parser.parse_args()
    config = load_config(args.config)

    # for loop over seeds
    for seed in range(args.num_seeds-1):
        seeded_config = config.copy()

        seeded_config['seed'] = seed
        seeded_config['experiment_name'] = f"{config['experiment_name']}_{seed}"
        seeded_config['model_path'] = f"{config['model_path']}_{seed}"
        
        # Call the main function with the updated config
        print(f"Running with seed: {seed}")

        main(seeded_config)