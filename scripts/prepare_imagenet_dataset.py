import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_utils import get_dataset, denormalize_image
from architectures import initialize_model
from robustness import attacker


def save_image(save, filename, image, dataset):
    if save:
        # Denormalize the first input image in the batch
        image = denormalize_image(image, dataset)  # Apply denormalization
            
        # Convert the tensor back to (H, W, C) format and scale to [0, 255]
        image_np = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)  # Scale and convert to uint8

        # convert to BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filename, image_np)

def setup_attack_kwargs():
    list_attacks = []
    for eps in [0.4, 0.2, 0.1, 0.05, 0.01]:
        attack_kwargs = {
            'constraint': "inf",  # L-inf PGD
            'eps': eps,  # Epsilon constraint (L-inf norm)
            'step_size': 0.01,  # Learning rate for PGD
            'iterations': 100,  # Number of PGD steps
            'targeted': False,  # Targeted attack
            'custom_loss': None  # Use default cross-entropy loss
        }
        list_attacks.append(attack_kwargs)
    return list_attacks

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare the ImageNet dataset for evaluation.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='Name of the dataset.')
    args = parser.parse_args()

    # General configuration
    dataset_name = args.dataset
    results_folder = "/media/pablo/Datasets/" + dataset_name

    # Setup attack parameters based on configuration
    list_attacks = setup_attack_kwargs()

    # Eval configuration
    n_samples = 10000

    # Create the results folder
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder + "/nat", exist_ok=True)
    for attack_kwargs in list_attacks:
        os.makedirs(results_folder + "/pgd" + str (attack_kwargs['eps']), exist_ok=True)

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the dataset and create data loaders
    dataset = get_dataset(dataset_name)
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=1)

    # Initialize the target model
    classifier, _, _ = initialize_model(1, dataset)

    # Initialize attacker
    attacker = attacker.Attacker(classifier.to(device), dataset).to(device)

    # Iterate over the test data loader with a progress bar
    sample_count = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on natural images", total=n_samples)):
        if sample_count >= n_samples:
            break
        inputs = dataset.normalize(inputs.to(device))

        save_image(True, results_folder + "/nat/"+str(batch_idx) + ".png", inputs[0].detach().cpu(), dataset)

        sample_count += 1

    for attack_kwargs in list_attacks:
        # Iterate over the test data loader again for adversarial examples
        sample_count = 0
        for batch_idx, (inputs, label) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on adversarial images with epsilon=" + str (attack_kwargs['eps']) , total=n_samples)):
            if sample_count >= n_samples:
                break
            inputs = inputs.to(device) # IMPORTANT: Should I normalize the inputs here?
            
            # Generate adversarial examples using the attacker
            target_label = (label + torch.randint_like(label, high=9)) % 10
            adv_im = attacker(inputs.to(device), target_label.to(device), True, **attack_kwargs)
            adv_im = dataset.normalize(adv_im)

            save_image(True, results_folder + "/pgd" + str (attack_kwargs['eps']) + "/"+str(batch_idx) + ".png", adv_im[0].detach().cpu(), dataset)

            sample_count += 1
