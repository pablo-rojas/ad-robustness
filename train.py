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
    i = 1
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
                results = val(detector, val_loader, device, norm, n_samples=config['train']['test_samples'])
                writer.add_scalar('Metrics/pAUC', results['pAUC'], i)

                if results['pAUC'] > best_pAUC:
                    best_pAUC = results['pAUC']
                    best_epoch = epoch
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
            best_epoch = epoch
            detector.save(save_path+'_best')
        
        # End training if the pAUC is more than 0.02 worse than the best pAUC, or it has not improved in the last 5 epochs
        if (results['pAUC'] < best_pAUC - 0.04) or (epoch - best_epoch > 10):
            print("Early stopping, best pAUC:", best_pAUC, "at epoch", best_epoch)
            break

        writer.add_scalar('Metrics/pAUC', results['pAUC'], i)
        print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")
        print(f"Epoch: {epoch}, pAUC: {results['pAUC']}, time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
        epoch += 1


def val(detector, val_loader, device, norm, n_samples=100, epsilon=0.03125):
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

    # Calculate the top 10% quantile 
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

def precompute_teacher_stats(detector, train_loader, norm, device, max_images=None):
    """
    Runs detector.teacher over up to max_images from train_loader,
    computes per-feature mean & std of its outputs, and writes them
    into detector.teacher_mean / teacher_std buffers.

    Args:
        detector: your UninformedStudents instance
        train_loader: DataLoader of anomaly‑free images
        norm: preprocessing transform
        device: torch device
        max_images: if not None, stop after this many images
    """
    detector.teacher.eval()
    sum_feats = None
    sumsq_feats = None
    total_vectors = 0
    images_seen = 0

    with torch.no_grad():
        for images, _ in train_loader:
            B = images.size(0)
            if max_images is not None and images_seen + B > max_images:
                # only take the remainder
                images = images[: max_images - images_seen]
                B = images.size(0)

            images = norm(images).to(device)
            feats = detector.teacher(images).detach()  # (B, d, H, W) or (B, d)
            # collapse spatial dims
            if feats.ndim > 2:
                B2, d, *sp = feats.shape
                feats = feats.permute(0, *range(2, feats.ndim), 1).reshape(-1, d)
            else:
                feats = feats.reshape(-1, feats.shape[-1])

            if sum_feats is None:
                # initialize on first batch
                sum_feats   = torch.zeros(feats.size(1), device=device)
                sumsq_feats = torch.zeros_like(sum_feats)

            sum_feats   += feats.sum(dim=0)
            sumsq_feats += (feats * feats).sum(dim=0)
            total_vectors += feats.size(0)
            images_seen   += B

            if max_images is not None and images_seen >= max_images:
                break

    # finalize
    mean = sum_feats / total_vectors
    var  = sumsq_feats / total_vectors - mean * mean
    std  = torch.sqrt(var.clamp(min=1e-6))

    # If the existing buffers are scalars, delete & re‑register them at the correct shape:
    if detector.teacher_mean.numel() == 1:
        # remove the old scalar buffers
        del detector._buffers['teacher_mean']
        del detector._buffers['teacher_std']
        # register new ones of the correct shape
        detector.register_buffer('teacher_mean', mean)
        detector.register_buffer('teacher_std',  std)
    else:
        # already the right shape: just overwrite
        detector.teacher_mean.data.copy_(mean)
        detector.teacher_std.data.copy_(std)

    print(f"[precompute] used {images_seen} images, {total_vectors} vectors")
    print(f"  mean  range: {mean.min():.4f} … {mean.max():.4f}")
    print(f"  std   range: {std.min():.4f} … {std.max():.4f}")


def main(config):
    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = UnifiedSummaryWriter()
    #writer.add_hparams(flatten_dict(config), {})
    #writer.add_text("Hyperparameters", json.dumps(config, indent=2))

    # Get the dataset and create data loaders
    dataset = get_dataset(config['dataset'])
    train_loader, val_loader, test_loader = dataset.make_loaders(batch_size=config['train']['batch_size'], workers=4)
    val_loader.dataset.ds_name = config['dataset']
    test_loader.dataset.ds_name = config['dataset']

    # Then call the function to initialize the detector
    detector = initialize_detector(config, dataset, device)

    # (Opotional) Precompute teacher stats if needed
    # if isinstance(detector, UninformedStudents):
    #     precompute_teacher_stats(detector, train_loader, dataset.normalize, device, max_images=10000)

    # Train the detector
    train(config, device, dataset.normalize, writer, train_loader, detector, val_loader)

    # Evaluate the detector
    results = val(detector, test_loader, device, dataset.normalize, n_samples=10000)

    # Log the hparams again with the updated metrics
    writer.add_hparams(flatten_dict(config), results)

    writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)