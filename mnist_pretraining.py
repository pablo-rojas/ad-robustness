import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from dataset_utils import get_mnist_loaders
from model_utils import ModifiedVGG11, DeepCNN
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create argument parser
parser = argparse.ArgumentParser(description='MNIST Pretraining')

# Add arguments
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

# Parse arguments
args = parser.parse_args()

# Hyper-parameters
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

# Model definition
model =  DeepCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

# Save the model checkpoint
torch.save(model.state_dict(), 'models/resnet18_mnist.pth')
