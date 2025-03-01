import torch
from torch import nn
import torchvision.models as models
from collections import OrderedDict

try:
    # If this is imported as a module (from src)
    from src.architectures import *
except ImportError:
    # If this is the main script
    from architectures import *


# Model weights paths based on dataset
model_paths = {
    'cifar': 'models/resnet18_cifar.pth',
    'mnist': 'models/resnet18_mnist.pth',
    'imagenet': None  # Uses torchvision's pretrained weights
}

def resnet18_classifier(device='cpu', dataset='imagenet', path=None, pretrained=True):
    """
    Constructs a ResNet18-based classifier model tailored for the specified dataset.
    Args:
        device (str): The compute device to allocate the model. Defaults to 'cpu'. Set to 'cuda' for GPU acceleration.
        dataset (str): The dataset type to configure the model. Supported values are:
            - 'cifar': Initializes a ResNet18 model for CIFAR datasets.
            - 'MNIST': Initializes a ResNet18 model for MNIST (grayscale images), modifying the input dimensions.
            - 'imagenet': Loads a pretrained ResNet18 model for ImageNet.
        path (str, optional): If provided, specifies a file path to load the model's state. For the 'MNIST' case, the state is expected under the 'net' key.
        pretrained (bool): Whether to load pretrained weights or initialize new model. Defaults to True.
    Returns:
        torch.nn.Module: The configured ResNet18-based classifier model, set to evaluation mode and moved to the specified device.
    Raises:
        ValueError: If an unsupported dataset type is provided.
    """

    if dataset=='cifar':
        model = ResNet18()
        if pretrained and path is not None:
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
        if pretrained and path is not None:
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
        if pretrained:
            model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            model = models.resnet18()

    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    model.to(device)
    model.eval()
    
    return model
    
# ...existing code...

def get_patch_descriptor(patch_size, dim=3):
    """
    Returns the appropriate patch descriptor model based on patch size.
    
    Args:
        patch_size (int): Size of the square patch (7, 17, 33, or 65)
        dim (int): Number of input channels, default is 3 for RGB images
        
    Returns:
        nn.Module: Appropriate patch descriptor model
        
    Raises:
        ValueError: If patch_size is not one of the supported sizes
    """
    if patch_size == 7:
        return Patch7Descriptor(dim=dim)
    elif patch_size == 17:
        return Patch17Descriptor(dim=dim)
    elif patch_size == 33:
        return Patch33Descriptor(dim=dim)
    elif patch_size == 65:
        return Patch65Descriptor(dim=dim)
    else:
        supported_sizes = [7, 17, 33, 65]
        raise ValueError(f"Unsupported patch size: {patch_size}. Supported sizes are: {supported_sizes}")
    
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
    
def initialize_us_models(num_students, dataset, patch_size, device='cpu'):
    """
    Initializes the teacher and student models for the given dataset.
    The teacher model uses pretrained weights via resnet18_classifier, while students have random weights.
    
    Args:
        num_students (int): The number of student models to initialize
        dataset: Dataset object or string indicating the dataset type ('cifar', 'mnist', or 'imagenet')
        device (str): Device to allocate models to ('cpu' or 'cuda')
        pretrained (bool): Whether to use pretrained weights for teacher model
    
    Returns:
        tuple: (teacher_model, teacher_feature_extractor, student_models)
            - teacher_model: Complete ResNet18 classifier
            - teacher_feature_extractor: ResNet18 without the fully connected layer
            - student_models: List of randomly initialized student models
    """
    # Determine dataset name if passed as object
    ds_name = dataset.ds_name if hasattr(dataset, 'ds_name') else dataset

    # Initialize teacher model using resnet18_classifier
    teacher_model = resnet18_classifier(device=device, dataset=ds_name, 
                                        path=model_paths[ds_name])
    
    # Create feature extractor (removing the fully connected layer)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:-1])
    teacher_feature_extractor.eval()
    
    # Initialize student models with random weights using Patch17Descriptor
    student_models = []
    for _ in range(num_students):
        student_model = get_patch_descriptor(patch_size=patch_size, dim=1 if ds_name == 'mnist' else 3)
        student_model.to(device)
        student_models.append(student_model)
    
    return teacher_model, teacher_feature_extractor, student_models

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models for each dataset
    datasets = ['cifar', 'mnist', 'imagenet']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Initializing model for {dataset.upper()} dataset")
        print(f"{'='*50}")
        
        # Get model path
        path = model_paths[dataset]
        print(f"Model path: {path if path else 'Using pretrained weights from torchvision'}")
        
        try:
            # Initialize the model
            model = resnet18_classifier(device=device, dataset=dataset)
            
            # Print model information
            print(f"Model type: {type(model).__name__}")
            
            # Print parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Create feature extractor (similar to the selected line)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            print(f"Feature extractor: " + str(feature_extractor))
            
        except Exception as e:
            print(f"Error initializing model for {dataset}: {e}")

if __name__ == "__main__":
    main()