import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from robustness import datasets
from sklearn.datasets import fetch_openml
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self):
        # Load MNIST dataset using scikit-learn
        mnist = fetch_openml('mnist_784', version=1)
        x, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

        # Normalize the images to [0, 1] and convert to float32
        x = x / 255.0
        self.data = x.reshape(-1, 28, 28).astype('float32')
        self.targets  = y.astype('int64')

        # Define transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Adjusted normalization
        ])
        self.transform = transform
        self.ds_name = 'mnist'  # Correctly set the dataset name
        self.mean = torch.tensor([0.5])
        self.std = torch.tensor([0.5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def make_loaders(self, workers=4, batch_size=100):
        train_size = int(0.8 * len(self.data))
        test_size = len(self.data) - train_size
        train_data, test_data = torch.utils.data.random_split(self, [train_size, test_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        return train_loader, test_loader
    
class CIFARDataset(datasets.CIFAR):
    def __init__(self, data_path='../cifar10-challenge/cifar10_data'):
        # Define transformations similar to MNISTDataset
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        super().__init__(data_path=data_path, transform_train=transform_train, transform_test=transform_test)
        self.ds_name = 'cifar'

def get_loaders(dataset, workers=4, batch_size=100):
    if dataset.ds_name == 'mnist':
        return dataset.make_loaders(workers=workers, batch_size=batch_size)
    elif dataset.ds_name == 'cifar':
        return dataset.make_loaders(workers=workers, batch_size=batch_size)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return MNISTDataset()
    elif dataset_name == 'cifar':
        return CIFARDataset()
    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')
