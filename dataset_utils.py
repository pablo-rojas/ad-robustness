import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from robustness import datasets as robustness_datasets
from sklearn.datasets import fetch_openml
import torchvision.datasets as datasets

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
            #transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to 3 channels
            transforms.ToTensor()
        ])
        self.transform = transform
        self.ds_name = 'mnist'  # Correctly set the dataset name
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

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
    
class CIFARDataset(robustness_datasets.CIFAR):
    def __init__(self, data_path='../cifar10-challenge/cifar10_data'):
        transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)  # CIFAR-10 normalization
        
        super().__init__(data_path=data_path, transform_train=transform, transform_test=transform)
        self.ds_name = 'cifar'

class ImageNetDataset:
    def __init__(self, data_path='path/to/imagenet'):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.data = datasets.ImageFolder(root=f"{data_path}/train", transform=transform)
        self.val_data = datasets.ImageFolder(root=f"{data_path}/val", transform=transform)
        self.ds_name = 'imagenet'
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def make_loaders(self, workers=4, batch_size=100):
        train_loader = DataLoader(self.data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        return train_loader, val_loader

def get_loaders(dataset, workers=4, batch_size=100):
    if dataset.ds_name == 'mnist':
        return dataset.make_loaders(workers=workers, batch_size=batch_size)
    elif dataset.ds_name == 'cifar':
        return dataset.make_loaders(workers=workers, batch_size=batch_size)
    elif dataset.ds_name == 'imagenet':
        return dataset.make_loaders(workers=workers, batch_size=batch_size)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

def get_dataset(dataset_name, data_path='/home/pablo/Datasets/ImageNet'):
    if dataset_name == 'mnist':
        return MNISTDataset()
    elif dataset_name == 'cifar':
        return CIFARDataset(data_path)
    elif dataset_name == 'imagenet':
        return ImageNetDataset(data_path)
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
