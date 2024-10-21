import argparse
import torch
from detector import Detector  # Import the Detector class
from robustness import attacker, datasets

from torch.utils.tensorboard import SummaryWriter
from model_utils import extract_patches
from eval_utils import partial_auc
import cv2
from tqdm import tqdm

if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Adversarial Example Generator')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon constraint (L-inf norm)')
    parser.add_argument('--linf', type=bool, default=False, help='L-inf constraint (True) or L2 (False)')
    parser.add_argument('--targeted', type=bool, default=False, help='Choose if the attack is targeted (True) or untargeted (False)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of PGD steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Use parsed arguments
    epsilon = args.epsilon
    linf = args.linf
    iterations = args.iterations
    targeted = args.targeted
    dataset_name = args.dataset
    epochs = args.epochs
    steps = 10000
    patch_size = 9
    save = False
    n_samples = 20 # Number of samples to evaluate

    # Setup attack parameters based on args
    if linf:
        attack_kwargs = {
            'constraint': 'inf',  # L-inf PGD
            'eps': epsilon,  # Epsilon constraint (L-inf norm)
            'step_size': 0.01,  # Learning rate for PGD
            'iterations': iterations,  # Number of PGD steps
            'targeted': targeted,  # Targeted attack
            'custom_loss': None  # Use default cross-entropy loss
        }
    else:
        attack_kwargs = {
            'constraint': '2',  # L2 PGD
            'eps': epsilon,  # Epsilon constraint (L2 norm)
            'step_size': 2.5 * epsilon / 100,  # Learning rate for PGD
            'iterations': iterations,  # Number of PGD steps
            'targeted': targeted,  # Targeted attack
            'custom_loss': None  # Use default cross-entropy loss
        }

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = SummaryWriter(comment=f"_{dataset_name}_l{attack_kwargs['constraint']}_{attack_kwargs['eps']}")

    # Initialize the detector model
    detector = Detector(10, dataset_name=dataset_name, patch_size=9)
    detector.load("models/detector_exp000")

    # Create data loaders
    train_loader, test_loader = detector.dataset.make_loaders(workers=4, batch_size=1)

    # Initialize attacker
    attacker = attacker.Attacker(detector.teacher, detector.dataset).to(device)

    # Initialize anomaly score list
    as_list = []
    e_list = []
    u_list = []
    nat_accuracy = 0

    # Iterate over the test data loader with a progress bar
    sample_count = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on natural images", total=n_samples)):
        if sample_count >= n_samples:
            break
        inputs = inputs.to(device)
        
        # Forward pass through the model to obtain the anomaly score
        regression_error, predictive_uncertainty = detector.forward(inputs)

        # Forward through the teacher model to obtain the prediction
        y = detector.teacher(detector.dataset.normalize(inputs)).detach().cpu()

        # Calculate the natural accuracy
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        # Append the values to the anomaly score list
        e_list.append(regression_error)
        u_list.append(predictive_uncertainty)

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

    # Log the top 1% quantiles to TensorBoard
    writer.add_scalar('Detector Evaluation/threshold', threshold, 0)

    # Log histograms of average and maximum standard deviations to TensorBoard
    writer.add_histogram('Histograms/anomaly_score', torch.tensor(as_list), 0)

    # Initialize lists to store average and maximum standard deviations for adversarial examples
    adv_as_list = []
    accuracy = 0
    adv_accuracy = 0

    # Iterate over the test data loader again for adversarial examples
    sample_count = 0
    for batch_idx, (inputs, label) in enumerate(tqdm(test_loader, desc="Evaluating the detector model on adversarial images", total=n_samples)):
        if sample_count >= n_samples:
            break
        inputs = inputs.to(device)
        
        # Generate adversarial examples using the attacker
        target_label = (label + torch.randint_like(label, high=9)) % 10
        adv_im = attacker(inputs.to(device), target_label.to(device), True, **attack_kwargs)

        # Forward through the teacher model to obtain the prediction
        y = detector.teacher(detector.dataset.normalize(adv_im)).detach().cpu()

        # Calculate the natural accuracy
        adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples
        
        # Forward pass through the model to obtain standard deviation map for adversarial examples
        regression_error, predictive_uncertainty = detector.forward(adv_im)

        # Calculate the anomaly score for adversarial examples
        anomaly_score = (regression_error - detector.e_mean) / detector.e_std + (predictive_uncertainty - detector.v_mean) / detector.v_std
        
        # Append the values to the anomaly score list
        adv_as_list.append(anomaly_score)

        # Count the number of adversarial examples that have higher standard deviations than the top 1% quantiles
        if anomaly_score > threshold:
            accuracy += 1

        if (save):
            cv2.imwrite(str(batch_idx) + "_adv.png", adv_im)

        sample_count += 1

    writer.add_scalar('Classifier Evaluation/natural_accuracy', nat_accuracy, 0)
    writer.add_scalar('Classifier Evaluation/adversarial_accuracy', adv_accuracy, 0)
    
    # Log the accuracy of adversarial examples to TensorBoard
    writer.add_scalar('Detector Evaluation/accuracy', accuracy/n_samples, 0)

    # Log histograms of average and maximum standard deviations for adversarial examples to TensorBoard
    writer.add_histogram('Histograms/adv_anomaly_score', torch.tensor(adv_as_list), 0)

    pAUC = partial_auc(as_list, adv_as_list)

    # Log the partial AUC to TensorBoard
    writer.add_scalar('Detector Evaluation/pAUC', pAUC, 0)