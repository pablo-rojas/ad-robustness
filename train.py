import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from src.detector import Detector
from src.model_utils import extract_patches
from src.dataset_utils import get_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/config.json', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration from JSON
    config = load_config(args.config)


    # Parameters from JSON
    dataset_name = config['dataset']
    save_path = "models/" + config['experiment_name']
    steps = config['train']['steps']
    patch_size = config['patch_size']

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = SummaryWriter(comment=f"_{dataset_name}")

    # Get the dataset and create data loaders
    dataset = get_dataset(dataset_name)
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=1)

    # Initialize the detector model
    detector = Detector(config['num_students'], dataset, patch_size=patch_size, device=device)

    i = 0
    # Train the model for the specified number of epochs
    with tqdm(total=steps, desc="Training Progress") as pbar:
        while i < steps:
            for batch_idx, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                patches = extract_patches(dataset.normalize(inputs), patch_size)

                # Reshape patches to have sufficient batch size
                patches = patches.view(-1, inputs.size(1), patch_size, patch_size)
                patches = patches.to(device)
                loss = detector.train_patch(patches)

                # Log the training loss
                writer.add_scalar('Loss/train', loss.item(), i)

                i += 1
                pbar.update(1)

                if i >= steps:
                    break

    detector.save(save_path)

