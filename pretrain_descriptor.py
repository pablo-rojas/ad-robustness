import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

from src.dataset_utils import get_dataset
from src.detector import *
from src.eval_utils import *
from src.model_utils import resnet18_classifier, model_paths
from src.misc_utils import *

from ACGAN.attacks.FGSM import FGSM


def pretrain(config, device, writer, train_loader, detector):
    
    # Parameters from JSON
    steps = config['train']['steps']

    # Initialize variables
    i = 1
    epoch = 1

    # Train the model for the specified number of epochs
    start_time = time.time()
    while i < steps:
        epoch_start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            loss = detector.train_step(images, labels)
            writer.add_scalar('Loss/train', loss.item(), i)
            
            i += 1
            if i >= steps:
                break

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        steps_left = steps - i
        estimated_time_left = (steps_left * elapsed_time) / i if i > 0 else 0


        print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")
        print(f"Epoch: {epoch},  time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
        epoch += 1


def main(config):
    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = UnifiedSummaryWriter()

    # Get the dataset and create data loaders
    dataset = get_dataset(config['dataset'], random_crop_size=config['patch_size']*2)
    train_loader, _ = dataset.make_loaders(batch_size=config['train']['batch_size'], workers=4, only_train=True)

    # Then call the function to initialize the detector
    detector = initialize_detector(config, dataset, device)

    # Train the detector
    pretrain(config, device, writer, train_loader, detector)

    # Save the model
    detector.save_pretrain(config['model_path'])

    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/pretrain.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)