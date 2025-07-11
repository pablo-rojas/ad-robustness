#!/usr/bin/env python3
import os
import json
import torch
import argparse
import random
import numpy as np

# Import necessary libraries for data loading
from torch.utils.data import Sampler

# Import your dataset class from your own code
from src.dataset_utils import get_dataset
from src.model_utils import resnet18_classifier
from src.eval_utils import partial_auc, get_target, sd_statistic

# Import the CNN and ACGAN definitions from the ACGAN folder
from ACGAN.GAN.acgan_1 import ACGAN
from ACGAN.GAN.acgan_res import ACGAN_Res
from ACGAN.attacks.FGSM import FGSM

class FixedOrderSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def test(gan, test_loader, device, n_samples=400, epsilon=0.05, targeted=1):
    target_model = resnet18_classifier(device, test_loader.dataset.ds_name)

    nat_as = []  # Will store anomaly scores for clean images.
    adv_as = []  # Will store anomaly scores for adversarial images.
    nat_accuracy = 0
    adv_accuracy = 0

    # Initialize the attacker
    fgsm = FGSM(model=target_model, norm=dataset.normalize, epsilon=epsilon, targeted=targeted)

    processed = 0
    for images, labels in test_loader:
        with torch.no_grad():
            # For classification, move inputs to GPU.
            y = target_model(dataset.normalize(images.to(device))).detach().cpu()

            # Choose target labels depending on the attack type.
            if targeted == 1: target_labels = get_target(labels)
            elif targeted == -1: target_labels = labels
            else:
                raise ValueError("Invalid value for targeted parameter. Must be 1 or -1.")

        # Generate adversarial images using the FGSM attack.
        adv_images = fgsm.attack(images.to(device), target_labels.to(device))

        with torch.no_grad():
            # Compute classification accuracy on adversarial images (using the classifier on GPU).
            y_adv = target_model(dataset.normalize(adv_images).to(device)).detach().cpu()

            # Check: Calculate natural and adversarial accuracy (Note that they should be 100% and 0% respectively).
            nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples
            adv_accuracy += (y_adv.argmax(1) == labels).sum().item()/n_samples

            # Calculate AS
            anomaly_score = sd_statistic(gan.discriminator(images.to(device)), labels)
            nat_as = np.append(nat_as, anomaly_score.item())

            anomaly_score_adv = sd_statistic(gan.discriminator(adv_images.to(device)), y_adv.argmax(1))
            adv_as = np.append(adv_as, anomaly_score_adv.item())

        
        if processed >= n_samples:
            break

        

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()
    detected = np.sum(adv_as > threshold)
    acc = (detected / len(adv_as) * 100) if len(adv_as) > 0 else 0

    # Compute partial AUC (pAUC) using the provided utility.
    pAUC = partial_auc(nat_as.tolist(), adv_as.tolist())

    # -------------------------------s
    # 4. Save and display results.
    # -------------------------------
    results = {
        "threshold": threshold,
        "natural_accuracy": nat_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "adversarial_detection_accuracy": acc,
        "pAUC": pAUC
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACGAN using a custom dataset")
    parser.add_argument('--dataset', type=str, default='svhn', 
                        help='Dataset to use for training (default: cifar)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate for the optimizer (default: 0.0002)')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train the model (default: 90)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')

    args = parser.parse_args()              
    dataset_name = args.dataset.lower()
    if args.seed is None:
        experiment_name = dataset_name + '_acgan'
        seed = 42  # Default seed if not provided
    else:
        experiment_name = dataset_name + '_acgan_' + str(args.seed)
        seed = args.seeds
        
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create directory for saving the trained model
    save_path = os.path.join("models", experiment_name)
    os.makedirs(save_path, exist_ok=True)

    # Load your dataset using your own dataset class; fixed batch size=100.
    dataset = get_dataset(dataset_name)
    train_loader, val_loader, test_loader = dataset.make_loaders(workers=4, batch_size=args.batch_size, seed=seed)

    test_loader.dataset.ds_name = dataset_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note that this CNN model not used at all in the ACGAN training (unless acgan_features is used instead of acgan_1 or acgan_res).
    model = resnet18_classifier(device=device, dataset=dataset_name)
    im_channel = 3  # resnet18 expects 3-channel images
    in_dim = 500
    if dataset_name == 'imagenet':
        num_classes = 1000
    else:
        # For MNIST, and CIFAR10, we can use 10 classes.
        # Adjust this based on your dataset.
        num_classes = 10

    if (dataset_name == 'mnist'):
         gan = ACGAN(
        in_dim=50,
        class_dim=num_classes,
        g_filters=[384, 192, 96, 48, im_channel],
        g_strides=[1, 2, 2, 2],
        d_filters=[16, 32, 64, 128, 256, 512],
        d_strides=[2, 1, 2, 1, 2, 1],
        CNN=model
    )
    else:
        gan = ACGAN_Res(in_dim=in_dim, class_dim=num_classes, CNN=model)
   
    print("Starting ACGAN training ...")
    # The gan.train() method internally loops over epochs and batches.
    gan.train(train_loader, lr=args.lr, num_epochs=args.epochs)

    # Save the generator and discriminator models after training.
    gan.save_model(save_path)
    print(f"Models saved to {save_path}")

    gan.load_model(save_path)
    gan.generator.eval()
    gan.discriminator.eval()

    # Test the trained ACGAN model
    results = test(gan, test_loader, device)
    print(results)
