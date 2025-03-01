import torch
from torch import nn
import torchvision.models as models
from src.architectures import *
from collections import OrderedDict


def resnet18_classifier(device='cpu', dataset='imagenet', path=None):
    """
    Constructs a ResNet18-based classifier model tailored for the specified dataset.
    Args:
        device (str): The compute device to allocate the model. Defaults to 'cpu'. Set to 'cuda' for GPU acceleration.
        dataset (str): The dataset type to configure the model. Supported values are:
            - 'cifar': Initializes a ResNet18 model for CIFAR datasets.
            - 'MNIST': Initializes a ResNet18 model for MNIST (grayscale images), modifying the input dimensions.
            - 'imagenet': Loads a pretrained ResNet18 model for ImageNet.
        path (str, optional): If provided, specifies a file path to load the model's state. For the 'MNIST' case, the state is expected under the 'net' key.
    Returns:
        torch.nn.Module: The configured ResNet18-based classifier model, set to evaluation mode and moved to the specified device.
    Raises:
        ValueError: If an unsupported dataset type is provided.
    """

    if dataset=='cifar':
        model = ResNet18()
        if path is not None:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            state_dict = checkpoint['net']
            if next(iter(state_dict)).startswith("module."):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                state_dict = new_state_dict

            model.load_state_dict(state_dict)

    elif dataset=='mnist':
        model = ResNet18(dim=1)
        if path is not None:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            state_dict = checkpoint['net']

            if next(iter(state_dict)).startswith("module."):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                state_dict = new_state_dict

            model.load_state_dict(state_dict)

    elif dataset=='imagenet':
        model = models.resnet18(pretrained=True, weights_only=True)

    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    model.to(device)
    model.eval()
    
    return model
    
    
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

def singe_discriminator_statistic(discriminator_output, target_label):
    """
    Converts the output of the discriminator into a single statistic, as described in the paper.
    On top of that, for simplicity of comparisson, I changed the sign of the output to be positive, 
    and converted all negative results to 100 (a very large number)

    Args:
        discriminator_output (torch.Tensor, torch.Tensor): Output of the discriminator.
        target_label (int): Target label of the attack.

    Returns:
        torch.Tensor: The extracted patches.
    """
    aux_prob, aux_out = discriminator_output
    s_d = torch.log(aux_prob) + torch.log(aux_out[:, target_label])
    if torch.isneginf(s_d).any():
        return torch.tensor(-100.0, device=s_d.device)
    return -s_d


def initialize_model_cifar(num_students, dataset, resume_path='models/cifar_nat.pt'):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """

    # Define the model architecture
    teacher_model = models.resnet18(pretrained=False)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # CIFAR-10 has 10 classes

    # Load the model weights
    teacher_model.load_state_dict(torch.load('models/resnet18_cifar.pth'))
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    # aux_model = models.resnet18(pretrained=True)
    # teacher_feature_extractor = torch.nn.Sequential(*list(aux_model.children())[:8])
    # teacher_feature_extractor.eval()

    # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = Patch17Descriptor()
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model_mnist(num_students, dataset, resume_path='models/mnist_nat.pt'):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """
    # Model definition
    teacher_model = models.resnet18(pretrained=False)
    num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_ftrs, 10)  # Adjust the final layer for MNIST/CIFAR10
    teacher_model = teacher_model

    # Load the model checkpoint
    teacher_model.load_state_dict(torch.load('models/resnet18_mnist.pth'))

    # Set the model to evaluation mode
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    aux_model = models.resnet18(pretrained=True)
    teacher_feature_extractor = torch.nn.Sequential(*list(aux_model.children())[:8])
    teacher_feature_extractor.eval()

     # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = Patch17Descriptor()
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model_imagenet(num_students, dataset):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """
    # Model definition
    teacher_model = models.resnet18(pretrained=True)

    # Set the model to evaluation mode
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.eval()

     # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = Patch17Descriptor()
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model(num_students, dataset):
    """
    Initializes the teacher and student models. All models will use the same architecture, but the teacher will use 
    pretrained weights, and the students will have random weights. The three versions of the model will be: one 
    teacher with the complete classifier (including the fully connected layer), one teacher without the fully 
    connected layer (feature extractor only), and a bag of students (also feature extractor only).

    Args:
        num_students (int): The number of student models.
        resume_path (str, optional): The path to a checkpoint to resume from. Defaults to 'models/cifar_linf_8.pt'.

    Returns:
        torch.nn.Module, torch.nn.Module, list: The teacher model, teacher feature extractor, and student models.
    """
    if dataset.ds_name == 'cifar':
        return initialize_model_cifar(num_students, dataset)
    elif dataset.ds_name == 'mnist':
        return initialize_model_mnist(num_students, dataset)
    elif dataset.ds_name == 'imagenet':
        return initialize_model_imagenet(num_students, dataset)
    else:
        raise ValueError(f"Dataset {dataset.ds_name} not supported")