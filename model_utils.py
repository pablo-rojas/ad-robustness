import torch
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
    def __init__(self, num_channels=3):
        super(ShallowNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
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

# Modify VGG11 for MNIST
class ModifiedVGG11(nn.Module):
    def __init__(self):
        super(ModifiedVGG11, self).__init__()
        self.model = models.vgg11(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Change input channels to 1
        self.model.classifier[6] = nn.Linear(4096, 10)  # Change output classes to 10
    
    def forward(self, x):
        return self.model(x)

# Define a custom deep CNN for MNIST
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU())
            #nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

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

    # Define the model architecture
    teacher_model = models.resnet18(pretrained=False)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    teacher_model = teacher_model.to(device)
    teacher_model.to(device)

    # Load the model weights
    teacher_model.load_state_dict(torch.load('models/resnet18_cifar.pth'))
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.to(device)
    teacher_feature_extractor.eval()

    # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = ShallowNet(num_channels=3).to(device)
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model_mnist(num_students, dataset, device, resume_path='models/mnist_nat.pt'):
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
    # Model definition
    teacher_model = models.resnet18(pretrained=False)
    num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_ftrs, 10)  # Adjust the final layer for MNIST/CIFAR10
    teacher_model = teacher_model.to(device)

    # Load the model checkpoint
    teacher_model.load_state_dict(torch.load('models/resnet18_mnist.pth'))

    # Set the model to evaluation mode
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.to(device)
    teacher_feature_extractor.eval()

     # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = ShallowNet(num_channels=3).to(device)
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

def initialize_model_imagenet(num_students, dataset, device):
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
    # Model definition
    teacher_model = models.resnet18(pretrained=True)
    teacher_model = teacher_model.to(device)

    # Set the model to evaluation mode
    teacher_model.eval()

    # Create the teacher feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:8])
    teacher_feature_extractor.to(device)
    teacher_feature_extractor.eval()

     # Initialize student models with random weights using the shallow network
    student_models = []
    for _ in range(num_students):
        student_model = ShallowNet(num_channels=3).to(device)
        student_models.append(student_model)

    return teacher_model, teacher_feature_extractor, student_models

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
    elif dataset.ds_name == 'imagenet':
        return initialize_model_imagenet(num_students, dataset, device)
    else:
        raise ValueError(f"Dataset {dataset.ds_name} not supported")

