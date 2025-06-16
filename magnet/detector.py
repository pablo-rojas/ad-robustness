import torch
import torch.nn as nn

from magnet.defensive_models import DefensiveModel1, DefensiveModel2
from magnet.evaluate_defensive_model import compute_reconstruction_error
from magnet.train_defensive_model import train_epoch, test

class MagNetDetector(nn.Module):
    """
    A wrapper class for the MagNet defensive models.
    This class allows for easy switching between different defensive models.
    It only implements the detection part of the MagNet framework, not 
    the reformer part.
    """

    def __init__(self, model, num_classes, device="cuda", dataset=None):
        super(MagNetDetector, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.dataset = dataset.ds_name

        if self.dataset == 'mnist':
            self.model = DefensiveModel1(in_channels=1).to(self.device)
        elif self.dataset == 'cifar-10':
            self.model = DefensiveModel2(in_channels=3).to(self.device)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
    def save(self, path):
        """
        Save the model to the specified path.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load the model from the specified path.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)

    def fit(self, train_loader, val_loader=None, num_epochs=10, lr=0.001, weight_decay=0.0001):
        """
        Fit the model to the training data.
        """
        pass

    def forward(self, x):
        """
        Forward pass through the defensive model.
        :param x: Input tensor
        :return: Output tensor after passing through the defensive model
        """
        x = self.model(x)
        return x

        