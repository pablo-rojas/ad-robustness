import torch
import torchvision.models as models
from collections import OrderedDict

try:
    # If this is imported as a module (from src)
    from src.architectures import *
    from src.wideresnet import DenseWideResNet, WideResNet
except ImportError:
    # If this is the main script
    from architectures import *


# Model weights paths based on dataset
model_paths = {
    'cifar': 'models/resnet18_cifar.pth',
    'mnist': 'models/resnet18_mnist.pth',
    'svhn': 'models/resnet18_svhn.pth',
    'imagenet': None  # Uses torchvision's pretrained weights
}

def resnet18_classifier(device='cpu', dataset='imagenet', path=None, pretrained=True):
    """
    Constructs a ResNet18-based classifier model tailored for the specified dataset.
    Args:
        device (str): The compute device to allocate the model. Defaults to 'cpu'. Set to 'cuda' for GPU acceleration.
        dataset (str): The dataset type to configure the model. Supported values are:
            - 'cifar': Initializes a ResNet18 model for CIFAR datasets.
            - 'mnist': Initializes a ResNet18 model for MNIST (grayscale images), modifying the input dimensions.
            - 'imagenet': Loads a pretrained ResNet18 model for ImageNet.
        path (str, optional): If provided, specifies a file path to load the model's state. For the 'MNIST' case, the state is expected under the 'net' key.
        pretrained (bool): Whether to load pretrained weights or initialize new model. Defaults to True.
    Returns:
        torch.nn.Module: The configured ResNet18-based classifier model, set to evaluation mode and moved to the specified device.
    Raises:
        ValueError: If an unsupported dataset type is provided.
    """

    if dataset=='cifar':
        if 'wideresnet' in (path or '').lower():
            model = WideResNet(depth=94,
                   num_classes=10,
                   widen_factor=16,
                   dropRate=0.0)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)
            

        else:

            model = ResNet18()
            if pretrained and path is not None:
                checkpoint = torch.load(path, map_location=device, weights_only=True)

            if 'net' in checkpoint:
                state_dict = checkpoint['net']
                if next(iter(state_dict)).startswith("module."):
                    new_state_dict = OrderedDict()
                    for key, value in state_dict.items():
                        new_key = key.replace("module.", "")
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                model.load_state_dict(state_dict)
            else:
                raw_sd = checkpoint['model']
            
                fixed_sd = {}
                for k, v in raw_sd.items():
                    name = k
                    if name.startswith('module.'):
                        name = name[len('module.'):]
                    if name.startswith('model.'):
                        name = name[len('model.'):]
                    fixed_sd[name] = v
                model_keys = set(model.state_dict().keys())
                state_dict = {k: v for k, v in fixed_sd.items() if k in model_keys}
                model.load_state_dict(state_dict)

    elif dataset=='mnist':
        model = ResNet18(dim=1)
        if pretrained and path is not None:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)

    elif dataset=='imagenet':
        if pretrained:
            model = ResNet18ImageNet()
        else:
            model = models.resnet18()

    elif dataset=='svhn':
        model = ResNet18(dim=3)
        if pretrained and path is not None:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)

    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    model.to(device)
    model.eval()
    
    return model

def resnet18_feature_extractor(device='cpu', dataset='imagenet', path="models/resnet18_imagenet.pth", freeze=True):
    """Creates and returns a ResNet18 feature extractor for different datasets.
    This function initializes a ResNet18-based feature extractor tailored for specific datasets
    (CIFAR, MNIST, or ImageNet). It loads pre-trained weights from the specified path and
    optionally freezes the model parameters for feature extraction.
    Args:
        device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        dataset (str, optional): Target dataset type. Supported values are 'cifar', 'mnist', 
            and 'imagenet'. Defaults to 'imagenet'.
        path (str, optional): Path to the pre-trained model weights file. Defaults to 
            "models/resnet18_imagenet.pth".
        freeze (bool, optional): Whether to freeze model parameters and set to evaluation mode.
            Defaults to True.
    Returns:
        torch.nn.Module: A ResNet18-based feature extractor model loaded with pre-trained weights
            and configured for the specified dataset.
    Raises:
        ValueError: If the specified dataset is not supported (not 'cifar', 'mnist', or 'imagenet').
    Note:
        - For MNIST dataset: Uses DenseCIFARResNet18 with single channel input.
        - For ImageNet dataset: Uses standard ResNet18 modified for dense feature extraction.
        - The function handles various checkpoint formats and automatically strips 'module.' 
          and 'model.' prefixes from state dictionary keys when needed. This has been done 
        to ensure compatibility with models trained from different sources.
    """
    
    if dataset=='cifar':
        if 'wideresnet' in (path or '').lower():
            teacher_feature_extractor = DenseWideResNet(depth=94,
                   num_classes=10,
                   widen_factor=16,
                   dropRate=0.0)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            teacher_feature_extractor.load_state_dict(checkpoint)
            

        else:
            teacher_feature_extractor = DenseCIFARResNet18(BasicBlock, [2, 2, 2, 2], dim=3).to(device) # Note how the use of this class allows to read the deafault resnet model, but changes the padding and stride, and skips the lineas and avgpool layers at the end

            checkpoint = torch.load(path, map_location=device, weights_only=True)
            # check if state_dict contains 'net' key
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
                if next(iter(state_dict)).startswith("module."):
                    new_state_dict = OrderedDict()
                    for key, value in state_dict.items():
                        new_key = key.replace("module.", "")
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                teacher_feature_extractor.load_state_dict(state_dict)
            else:
                raw_sd = checkpoint['model']
            
                fixed_sd = {}
                for k, v in raw_sd.items():
                    name = k
                    if name.startswith('module.'):
                        name = name[len('module.'):]
                    if name.startswith('model.'):
                        name = name[len('model.'):]
                    fixed_sd[name] = v
                model_keys = set(teacher_feature_extractor.state_dict().keys())
                state_dict = {k: v for k, v in fixed_sd.items() if k in model_keys}
                teacher_feature_extractor.load_state_dict(state_dict)
        

    elif dataset=='mnist':
        teacher_feature_extractor = DenseCIFARResNet18(BasicBlock, [2, 2, 2, 2], dim=1).to(device) # Note how the use of this class allows to read the deafault resnet model, but changes the padding and stride, and skips the lineas and avgpool layers at the end
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        teacher_feature_extractor.load_state_dict(checkpoint)

    elif dataset=='imagenet':
        resnet18 = models.resnet18(weights='IMAGENET1K_V1').to(device)
        resnet18 = modify_resnet_for_dense(resnet18)
        teacher_feature_extractor = DenseResNet18(resnet18)

    elif dataset=='svhn':
        teacher_feature_extractor = DenseCIFARResNet18(BasicBlock, [2, 2, 2, 2], dim=3).to(device) # Note how the use of this class allows to read the deafault resnet model, but changes the padding and stride, and skips the lineas and avgpool layers at the end
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        teacher_feature_extractor.load_state_dict(checkpoint)

    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    if freeze:
        # Freeze the feature extractor and set to evaluation mode
        for param in teacher_feature_extractor.parameters():
            param.requires_grad = False
        teacher_feature_extractor = teacher_feature_extractor.eval()
        teacher_feature_extractor.eval()
    

    return teacher_feature_extractor.to(device)

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
        tuple: (teacher, student)
            - teacher: ResNet18 without the fully connected layer
            - student: List of randomly initialized student models
    """
    # Determine dataset name if passed as object
    ds_name = dataset.ds_name if hasattr(dataset, 'ds_name') else dataset

    
    # Initialize teacher model with pretrained weights
    teacher = resnet18_feature_extractor(device, ds_name, model_paths[ds_name])
    # teacher = resnet18_feature_extractor(device, ds_name, 'models/wideresnet18_cifar_best.pth')

    # Initialize student models with random weights using Patch17Descriptor
    students = []
    for _ in range(num_students):
        student = get_patch_descriptor(patch_size=patch_size, dim=1 if ds_name == 'mnist' else 3)
        student.to(device)
        students.append(student)
    
    return teacher, students


