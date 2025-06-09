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

from mahalanobis.detector import MahalanobisDetector, LIDDetector
from ACGAN.attacks.FGSM import FGSM

def val(detector, val_loader, device, norm, n_samples=100, epsilon=0.01):
    dataset = val_loader.dataset
    target_model = resnet18_classifier(device, dataset.ds_name, path=model_paths[dataset.ds_name])

    nat_as = []  # Will store anomaly scores for clean images.
    adv_as = []  # Will store anomaly scores for adversarial images.    

    i = 0
    for images, labels in val_loader:
        images = norm(images.to(device))
        labels = labels.to(device)

        anomaly_score = detector(images).detach().cpu().numpy()
        nat_as.append(anomaly_score)

        i += 1
        if i >= n_samples:
            break

    nat_as = np.array(nat_as)

    # Calculate the top 10% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()

    # Initialize the attacker on untargeted mode
    fgsm = FGSM(model=target_model, norm=norm, epsilon=epsilon, targeted=-1)

    i = 0
    for images, labels in val_loader:
        adv_images = fgsm.attack(images.to(device), labels)

        anomaly_score = detector(norm(adv_images).to(device)).detach().cpu().numpy()
        adv_as.append(anomaly_score)

        i += 1
        if i >= n_samples:
            break

    adv_as = np.array(adv_as)
    detected = np.sum(adv_as > threshold)
    
    return {"det_acc": (detected / len(adv_as) * 100) if len(adv_as) > 0 else 0,
        "pAUC": partial_auc(nat_as.tolist(), adv_as.tolist())}

def main(config, type='mahalanobis', read_from_file=False):
    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    detector_path = f"models/mahalanobis_detector_{config['dataset']}"

    # Get the dataset and create data loaders
    dataset = get_dataset(config['dataset'], normalize=True)
    train_loader, _, _ = dataset.make_loaders(batch_size=1, workers=4, seed=seed)
    _, val_loader, test_loader = get_dataset(config['dataset']).make_loaders(batch_size=1, workers=4, seed=seed)
    test_loader.dataset.ds_name = config['dataset']
    norm = dataset.normalize
    num_classes = 1000 if config['dataset'] == 'imagenet' else 10

    target_model = resnet18_classifier(device, dataset.ds_name, path=model_paths[dataset.ds_name])

    # Then call the function to initialize the detector
    if type == 'lid':
        detector = LIDDetector(target_model, num_classes=num_classes, device='cuda', net_type='resnet', dataset=dataset)
    elif type == 'mahalanobis':
        detector = MahalanobisDetector(target_model, num_classes=num_classes, device='cuda', net_type='resnet', dataset=dataset)
    else:
        raise ValueError(f"Unknown detector type: {type}")

    if not read_from_file:
        detector.fit(train_loader)

        if type == 'mahalanobis':
            fgsm = FGSM(model=target_model, norm=norm, epsilon=0.05, targeted=-1)
            detector.train_regressor(val_loader, norm, fgsm)

        # Save the trained detector
        detector.save(detector_path)
    else:
        detector.load(detector_path)

    # Evaluate the detector
    results = val(detector, test_loader, device, norm, n_samples=500, epsilon=0.01)
    print(f"Detection Accuracy: {results['det_acc']:.2f}%")
    print(f"Partial AUC: {results['pAUC']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/imagenet_train_us.json', help='Path to the configuration file.')
    parser.add_argument('--type', type=str, default='mahalanobis', help='Type of detector to use. Options: mahalanobis, lid')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config, type=args.type)