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


def train(config, device, norm, writer, train_loader, detector, val_loader=None):
    
    # Parameters from JSON
    save_path = config['model_path']
    steps = config['train']['steps']
    val_interval = config['train']['test_interval'] if config['train']['test_interval'] != 0 else None

    # Initialize variables
    i = 0
    epoch = 1
    best_pAUC = 0
    best_epoch = 1

    # Train the model for the specified number of epochs
    start_time = time.time()
    while i < steps:
        epoch_start = time.time()
        for images, labels in train_loader:
            images = norm(images).to(device)
            labels = labels.to(device)

            loss = detector.train_step(images, labels)
            writer.add_scalar('Loss/train', loss.item(), i)

            if val_interval is not None and i % val_interval == 0:
                results = val(detector, val_loader, device, norm)
                writer.add_scalar('Metrics/pAUC', results['pAUC'], i)

                if results['pAUC'] > best_pAUC:
                    best_pAUC = results['pAUC']
                    detector.save(save_path+'_best')
                    print("Model saved at", save_path+'_best')
            
            i += 1
            if i >= steps:
                break

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        steps_left = steps - i
        estimated_time_left = (steps_left * elapsed_time) / i if i > 0 else 0
            
        results = val(detector, val_loader, device, norm, n_samples=config['train']['test_samples'])
        detector.save(save_path+'_last')

        # Save the model if the pAUC is better than the previous best
        if results['pAUC'] > best_pAUC:
            best_pAUC = results['pAUC']
            detector.save(save_path+'_best')
        
        # End training if the pAUC is more than 0.02 wors than the best pAUC, or it has not improved in the last 5 epochs
        if (results['pAUC'] < best_pAUC - 0.02) or (epoch - best_epoch > 5):
            print("Early stopping")
            break

        writer.add_scalar('Metrics/pAUC', results['pAUC'], i)
        print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")
        print(f"Epoch: {epoch}, pAUC: {results['pAUC']}, time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
        epoch += 1

def val(detector, val_loader, device, norm, n_samples=100, epsilon=0.05):
    dataset = val_loader.dataset
    target_model = resnet18_classifier(device, dataset.ds_name, path=model_paths[dataset.ds_name])

    e_list = []
    u_list = []
    nat_as = []  # Will store anomaly scores for clean images.
    adv_as = []  # Will store anomaly scores for adversarial images.

    i = 0
    for images, labels in val_loader:
        images = norm(images.to(device))
        labels = labels.to(device)

        if isinstance(detector, UninformedStudents):
            re, pu = detector.forward(images, labels)
            e_list.append(re)
            u_list.append(pu)
        else:
            anomaly_score = detector.forward(images)
            nat_as.append(anomaly_score.item())

        i += 1
        if i >= n_samples:
            break

    if isinstance(detector, UninformedStudents):
        # Calculate the mean and standard deviation of e_list and u_list
        detector.e_mean = torch.tensor(e_list).mean().item()
        detector.e_std = torch.tensor(e_list).std().item()
        detector.v_mean = torch.tensor(u_list).mean().item()
        detector.v_std = torch.tensor(u_list).std().item()

        # Normalize e_list and u_list and calculate the anomaly score
        nat_as = [(e - detector.e_mean) / detector.e_std + (u - detector.v_mean) / 
                  detector.v_std for e, u in zip(e_list, u_list)]
        
    nat_as = np.array(nat_as)

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()

    # Initialize the attacker on untargeted mode
    fgsm = FGSM(model=target_model, norm=norm, epsilon=epsilon, targeted=-1)

    i = 0
    for images, labels in val_loader:
        adv_images = fgsm.attack(images.to(device), labels)

        if isinstance(detector, UninformedStudents):
            re, pu = detector.forward(norm(adv_images).to(device)) # For conditional models I should pass the target model prediction
            anomaly_score = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
            adv_as.append(anomaly_score)
        else:
            anomaly_score = detector.forward(norm(adv_images).to(device))
            adv_as.append(anomaly_score.cpu().item())

        i += 1
        if i >= n_samples:
            break

    adv_as = np.array(adv_as)
    detected = np.sum(adv_as > threshold)
    
    return {"det_acc": (detected / len(adv_as) * 100) if len(adv_as) > 0 else 0,
        "pAUC": partial_auc(nat_as.tolist(), adv_as.tolist())}

def main(config):
    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = SummaryWriter()
    #writer.add_hparams(flatten_dict(config), {}, run_name='')
    writer.add_text("Hyperparameters", json.dumps(config, indent=2))

    # Get the dataset and create data loaders
    dataset = get_dataset(config['dataset'])
    train_loader, val_loader = dataset.make_loaders(batch_size=config['train']['batch_size'], workers=4, only_train=True)
    val_loader.dataset.ds_name = config['dataset']

    # Then call the function to initialize the detector
    detector = initialize_detector(config, dataset, device)

    # Train the detector
    train(config, device, dataset.normalize, writer, train_loader, detector, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)