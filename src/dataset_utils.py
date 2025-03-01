import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from robustness import datasets as robustness_datasets
from sklearn.datasets import fetch_openml
import torchvision.datasets as datasets
from torch.utils.data import Sampler

import numpy as np
import random

class FixedOrderSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
class BaseDataset(Dataset):
    def make_loaders(self, workers=4, batch_size=100, only_train=False, 
                     only_test=False, seed=42):
        # Create deterministic but shuffled indices
        train_indices = list(range(len(self.train_data)))
        val_indices = list(range(len(self.val_data))) if hasattr(self, 'val_data') else []
        test_indices = list(range(len(self.test_data)))
        
        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(train_indices)
        random.shuffle(val_indices) if val_indices else None
        random.shuffle(test_indices)
        generator = torch.Generator().manual_seed(seed)

        # Create the data loaders
        train_loader = DataLoader(
            self.train_data, batch_size=batch_size, 
            sampler=FixedOrderSampler(train_indices),
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=generator
        )
        
        val_loader = None
        if hasattr(self, 'val_data'):
            val_loader = DataLoader(
                self.val_data, batch_size=1, 
                sampler=FixedOrderSampler(val_indices),
                num_workers=workers, pin_memory=True,
                worker_init_fn=seed_worker, generator=generator
            )
        
        test_loader = DataLoader(
            self.test_data, batch_size=1, 
            sampler=FixedOrderSampler(test_indices),
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=generator
        )

        # Return the loaders based on the arguments
        if only_train: return train_loader, val_loader
        if only_test: return test_loader
        return (train_loader, val_loader, test_loader) if val_loader else (train_loader, test_loader)
class MNISTDataset(BaseDataset):
    def __init__(self, data_path='./data/mnist', seed=42, normalize_images=False):
        # Set dataset name and normalization parameters
        self.ds_name = 'mnist'
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        # Define transformations based on normalize_images flag
        transform_list = [transforms.ToTensor()]
        if normalize_images:
            transform_list.append(self.normalize)
        transform = transforms.Compose(transform_list)

        # Set seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load MNIST dataset using torchvision
        mnist_train = datasets.MNIST(root=data_path, train=True, 
                                    download=True, transform=transform)
        self.test_data = datasets.MNIST(root=data_path, train=False, 
                                      download=True, transform=transform)
        
        # Create validation split from training data
        train_size = int(0.8 * len(mnist_train))
        val_size = len(mnist_train) - train_size
        generator = torch.Generator().manual_seed(seed)
        self.train_data, self.val_data = torch.utils.data.random_split(
            mnist_train, [train_size, val_size], generator=generator)
        
class CIFARDataset(BaseDataset):
    def __init__(self, data_path='./data/cifar10', seed=42, normalize_images=False):
        # Set dataset name and normalization parameters needed for adv attacks
        self.ds_name = 'cifar'
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # Define transformations based on normalize_images flag
        transform_list = [transforms.ToTensor()]
        if normalize_images:
            transform_list.append(self.normalize)
        transform = transforms.Compose(transform_list)

        # Set seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load CIFAR-10 dataset using torchvision
        cifar_train = datasets.CIFAR10(root=data_path, train=True, 
                                      download=True, transform=transform)
        self.test_data = datasets.CIFAR10(root=data_path, train=False, 
                                         download=True, transform=transform)
        
        # Create validation split from training data
        train_size = int(0.8 * len(cifar_train))
        val_size = len(cifar_train) - train_size
        generator = torch.Generator().manual_seed(seed)
        self.train_data, self.val_data = torch.utils.data.random_split(
            cifar_train, [train_size, val_size], generator=generator)
        

class ImageNetDataset(BaseDataset):
    def __init__(self, data_path='./data/ImageNet', normalize_images=False):
        # Set dataset name and normalization parameters needed for adv attacks
        self.ds_name = 'imagenet'
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # Define transformations based on normalize_images flag
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
        if normalize_images:
            transform_list.append(self.normalize)
        transform = transforms.Compose(transform_list)

        # Load ImageNet dataset using torchvision
        self.train_data = datasets.ImageFolder(root=f"{data_path}/train", 
                                              transform=transform)
        self.val_data = datasets.ImageFolder(root=f"{data_path}/val", 
                                            transform=transform)
        self.test_data = datasets.ImageFolder(root=f"{data_path}/test", 
                                             transform=transform)

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return MNISTDataset()
    elif dataset_name == 'cifar':
        return CIFARDataset()
    elif dataset_name == 'imagenet':
        return ImageNetDataset()
    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')
    
# Define the denormalization function
def denormalize_image(image, dataset):
    """
    Reverses the normalization of an image based on the dataset's mean and std.
    
    Args:
    - image (torch.Tensor): Normalized image tensor of shape [C, H, W].
    - dataset (object): The dataset object containing the mean and std.
    
    Returns:
    - torch.Tensor: The denormalized image tensor.
    """
    mean = dataset.mean  # Mean from the dataset
    std = dataset.std    # Std from the dataset
    
    image = image.clone()  # Clone to avoid modifying original tensor
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # Reverse normalization: image * std + mean
    return image
