import argparse
import torch
from detector import Detector  # Import the Detector class
from torch.utils.tensorboard import SummaryWriter
from model_utils import extract_patches

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
    patch_size=9

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
    detector = Detector(10, attack_kwargs, dataset_name=dataset_name, patch_size=9)

    # Create data loaders
    train_loader, test_loader = detector.dataset.make_loaders(workers=4, batch_size=1)

    i = 0
    # Train the model for the specified number of epochs
    while i < steps:
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            patches = extract_patches(detector.dataset.normalize(inputs), patch_size)

            # Reshape patches to have sufficient batch size
            patches = patches.view(-1, inputs.size(1), patch_size, patch_size)
            patches = patches.to(device)
            loss = detector.train_patch(patches)

            # Log the training loss
            writer.add_scalar('Loss/train', loss.item(), i)

            i += 1

    detector.save_model("models/detector_exp000")

