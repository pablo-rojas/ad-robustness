import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np

from robustness import attacker, datasets
from robustness.model_utils import make_and_restore_model
from model_utils import extract_patches, initialize_model
from dataset_utils import get_dataset
import torchvision.models as models

def denormalize(tensor, mean, std):
    """Denormalize a tensor using the provided mean and std."""
    mean = mean[:, None, None]
    std = std[:, None, None]
    return tensor * std + mean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = get_dataset('imagenet')
print("Dataset: " + str(dataset))
teacher_model = models.resnet18(pretrained=True)
teacher_model = teacher_model.to(device)

# Set the model to evaluation mode
teacher_model.eval()
attacker = attacker.Attacker(teacher_model, dataset).to(device)
attack_kwargs = {
    'constraint': 'inf', # L-inf PGD 
    'eps': 0.1, # Epsilon constraint (L-inf norm)
    'step_size': 0.01, # Learning rate for PGD
    'iterations': 100, # Number of PGD steps
    'targeted': False, # Targeted attack
    'custom_loss': None # Use default cross-entropy loss
}


train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=1)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

for batch_idx, (inputs, label) in enumerate(test_loader):
    inputs = inputs.to(device)

    # Generate adversarial examples using the attacker
    target_label = (label + torch.randint_like(label, high=9)) % 10
    adv_im = attacker(inputs, target_label.to(device), True, **attack_kwargs)
    print('adv_im range:', adv_im.min().item(), adv_im.max().item())
    print('inputs range:', inputs.min().item(), inputs.max().item())

    # Forward through the teacher model to obtain the prediction
    y_adv = teacher_model(adv_im).detach().cpu()
    y = teacher_model(inputs).detach().cpu()
    print('y_adv:', y_adv.argmax(1))
    print('y:', y.argmax(1))
    print('label:', label)
    print('                   ------------------                  ')
    print('-------------------------------------------------------')

    diff = adv_im - inputs

    print('diff range:', diff.min().item(), diff.max().item())

    # Denormalize the images
    adv_im_denorm = adv_im[0].detach().cpu().numpy().transpose(1, 2, 0)
    inputs_denorm = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)

    # Convert to [0, 255] range
    adv_im_denorm = np.clip(adv_im_denorm * 255.0, 0, 255).astype(np.uint8)
    inputs_denorm = np.clip(inputs_denorm * 255.0, 0, 255).astype(np.uint8)

    # Save the images
    cv2.imwrite('adv_im.bmp', adv_im_denorm)
    cv2.imwrite('inputs.bmp', inputs_denorm)

    break

            
