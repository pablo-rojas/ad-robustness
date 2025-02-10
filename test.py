import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from src.dataset_utils import get_dataset, denormalize_image
from src.detector import Detector  # Import the Detector class
from robustness import attacker
from src.eval_utils import partial_auc, save_results
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


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

def setup_attack_kwargs(config):
    attack_kwargs = {
        'constraint': config['test']['attacker']['constraint'],  # L-inf PGD
        'eps': config['test']['attacker']['epsilon'],  # Epsilon constraint (L-inf norm)
        'step_size': config['test']['attacker']['step_size'],  # Learning rate for PGD
        'iterations': config['test']['attacker']['iterations'],  # Number of PGD steps
        'targeted': config['test']['attacker']['targeted'],  # Targeted attack
        'custom_loss': None  # Use default cross-entropy loss
    }
    return attack_kwargs

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/cifar_config.json', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration from JSON
    config = load_config(args.config)

    # General configuration
    dataset_name = config['dataset']
    save_path = "models/" + config['experiment_name']
    results_folder = "results/" + config['experiment_name']
    patch_size = config['patch_size']

    # Eval configuration
    n_samples = config['test']['samples']
    save = config['test']['save']

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder + "/img", exist_ok=True)

    # Setup attack parameters based on configuration
    attack_kwargs = setup_attack_kwargs(config)

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the dataset and create data loaders
    dataset = get_dataset(dataset_name)
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=1)

    # Initialize the detector model
    detector = Detector(config['num_students'], dataset, patch_size=patch_size, device=device)
    detector.load(save_path)

    # Initialize attacker
    attacker = attacker.Attacker(detector.teacher, dataset).to(device)

    # Initialize anomaly score list
    as_list = []
    e_list = []
    u_list = []
    nat_accuracy = 0

    # Iterate over the test data loader with a progress bar
    sample_count = 0
    for batch_idx, (inputs, label) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on natural images", total=n_samples)):
        if sample_count >= n_samples:
            break
        inputs = dataset.normalize(inputs.to(device))
        
        # Forward pass through the model to obtain the anomaly score
        regression_error, predictive_uncertainty = detector.forward(inputs)

        # Forward through the teacher model to obtain the prediction
        y = detector.teacher(inputs).detach().cpu()

        # Calculate the natural accuracy
        nat_accuracy += (y.argmax(1) == label).sum().item()/n_samples
 
        # Append the values to the anomaly score list
        e_list.append(regression_error)
        u_list.append(predictive_uncertainty)

        save_image(save, results_folder + "/img/"+str(batch_idx) + "_nat.png", inputs[0].detach().cpu(), dataset)

        sample_count += 1
    
    detector.e_mean = torch.tensor(e_list).mean().item()
    detector.e_std = torch.tensor(e_list).std().item()
    detector.v_mean = torch.tensor(u_list).mean().item()
    detector.v_std = torch.tensor(u_list).std().item()

    # Normalize e_list and u_list
    e_list = [(e - detector.e_mean) / detector.e_std for e in e_list]
    u_list = [(u - detector.v_mean) / detector.v_std for u in u_list]

    # Compute the anomaly score list
    as_list = [e + u for e, u in zip(e_list, u_list)]

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(as_list), 0.90).item()

    # Initialize lists to store average and maximum standard deviations for adversarial examples
    adv_as_list = []
    accuracy = 0
    adv_accuracy = 0

    # Iterate over the test data loader again for adversarial examples
    sample_count = 0
    for batch_idx, (inputs, label) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on adversarial images", total=n_samples)):
        if sample_count >= n_samples:
            break
        inputs = inputs.to(device) # IMPORTANT: Should I normalize the inputs here?
        
        # Generate adversarial examples using the attacker
        target_label = (label + torch.randint_like(label, high=9)) % 10
        adv_im = attacker(inputs.to(device), target_label.to(device), True, **attack_kwargs)
        adv_im = dataset.normalize(adv_im)

        # Forward through the teacher model to obtain the prediction
        y = detector.teacher(adv_im).detach().cpu()

        # Calculate the natural accuracy
        adv_accuracy += (y.argmax(1) == label).sum().item()/n_samples
        
        # Forward pass through the model to obtain standard deviation map for adversarial examples
        regression_error, predictive_uncertainty = detector.forward(adv_im)

        # Calculate the anomaly score for adversarial examples
        anomaly_score = (regression_error - detector.e_mean) / detector.e_std + (predictive_uncertainty - detector.v_mean) / detector.v_std
        
        # Append the values to the anomaly score list
        adv_as_list.append(anomaly_score)

        # Count the number of adversarial examples that have higher standard deviations than the top 1% quantiles
        if anomaly_score > threshold:
            accuracy += 1

        save_image(save, results_folder + "/img/"+str(batch_idx) + "_adv.png", adv_im[0].detach().cpu(), dataset)

        sample_count += 1

    results = {
        "nat_list": as_list,
        "adv_list": adv_as_list,
        "threshold": threshold,
        "natural_accuracy": nat_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "adversarial_detection_accuracy": accuracy / n_samples,
        "pAUC": partial_auc(as_list, adv_as_list)
    }
    save_results(results_folder, results)