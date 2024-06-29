import torch
import os
import dill
from torch import nn

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

def initialize_model(num_students, device, resume_path='models/cifar_linf_8.pt'):
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
    # Initialize teacher model with pretrained weights
    teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    # Optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path, pickle_module=dill)
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if 'model' not in checkpoint:
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        # Adjusting for the potential 'module.' prefix in model keys when using DataParallel
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        try:
            teacher_model.load_state_dict(sd)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)
        
    # Freeze teacher's layers
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_feature_extractor = nn.Sequential(*list(teacher_model.children())[:-1])  # Use as a feature extractor

    # Initialize student models with random weights
    student_models = []
    for _ in range(num_students):
        student_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        student_model.fc = nn.Identity()  # Use as a feature extractor
        student_models.append(student_model.to(device))

    return teacher_model.to(device), teacher_feature_extractor.to(device), student_models