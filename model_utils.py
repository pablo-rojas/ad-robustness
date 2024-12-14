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
    
class Patch17Descriptor(nn.Module):
    def __init__(self):
        super(Patch17Descriptor, self).__init__()

        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        
        x = self.leaky_relu(self.conv2(x))
        
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = self.decode(x)
        
        return x

class Patch33Descriptor(nn.Module):
    def __init__(self):
        super(Patch33Descriptor, self).__init__()

        # Architecture for p = 33
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1)
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.leaky_relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = self.decode(x)
        
        return x

class Patch65Descriptor(nn.Module):
    def __init__(self):
        super(Patch65Descriptor, self).__init__()
        
        # Define layers as per the table
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)

        # Activation function (Leaky ReLU with slope 5e-3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.leaky_relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.leaky_relu(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        
        x = self.decode(x)
        
        return x

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

