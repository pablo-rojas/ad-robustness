import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from dataset_utils import get_dataset, get_loaders 

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return test_loss / len(test_loader), correct / len(test_loader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args.dataset)
    train_loader, test_loader = get_loaders(dataset, workers=args.workers, batch_size=args.batch_size)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Adjust the final layer for MNIST/CIFAR10
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'models/resnet18_{args.dataset}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 on MNIST or CIFAR datasets')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], required=True, help='Dataset to use: mnist or cifar')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
