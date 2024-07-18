import torch
import os
import dill
from torch import nn
from robustness.attacker import AttackerModel
import torchvision.models as models

def extract_patches(image, patch_size):
    """
    Extracts patches from an image.

    Args:
        image (torch.Tensor): The input image.
        patch_size (int): The size of the patches.

    Returns:
        torch.Tensor: The extracted patches.
    """
    # Unfold to extract sliding local blocks
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # Combine the patches into a batch dimension
    patches = patches.contiguous().view(image.size(0), image.size(1), -1, patch_size, patch_size)

    # Permute to have patch dimension first
    patches = patches.permute(2, 0, 1, 3, 4)

    return patches

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def initialize_model_cifar(num_students, dataset, device, resume_path='models/cifar_nat.pt'):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        device (str): The device to use for training.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """

    # Load the teacher model with pretrained weights for CIFAR10
    teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher_model.to(device)
    teacher_model.eval()

    # Load the model from a checkpoint if resume_path is provided
    # if resume_path:
    #     checkpoint = torch.load(resume_path, map_location=device)
    #     teacher_model.load_state_dict(checkpoint['model_state_dict'])

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.to(device)
    teacher_feature_extractor.eval()

    # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = ShallowNet().to(device)
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model_mnist(num_students, dataset, device, resume_path='models/mnist_nat.pt'):

    # To do
    pass

def initialize_model(num_students, dataset, device):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        device (str): The device to use for training.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """
    if dataset.ds_name == 'cifar':
        return initialize_model_cifar(num_students, dataset, device)
    elif dataset.ds_name == 'mnist':
        return initialize_model_mnist(num_students, dataset, device)
    else:
        raise ValueError(f"Dataset {dataset.ds_name} not supported")

