import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import fetch_openml
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

def get_mnist_loaders(batch_size=100):
    # Load MNIST dataset using scikit-learn
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()

    # Normalize the images to [0, 1] and convert to float32
    x = x / 255.0
    x = x.reshape(-1, 28, 28).astype('float32')
    y = y.astype('int64')

    # Split into training and test sets
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create datasets
    train_dataset = MNISTDataset(x_train, y_train, transform=transform)
    test_dataset = MNISTDataset(x_test, y_test, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader