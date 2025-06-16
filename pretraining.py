import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src.dataset_utils import get_dataset
from src.model_utils import resnet18_classifier
from src.wideresnet import WideResNet

def train(model, train_loader, criterion, optimizer, trans, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = trans(inputs.to(device)), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, test_loader, criterion, norm, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = norm(inputs.to(device)), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return test_loss / len(test_loader), correct / len(test_loader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args.dataset)
    norm = dataset.normalize
    if args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            norm
            ])
    else:
        transform_train = transforms.Compose([norm])

    train_loader, val_loader, test_loader = dataset.make_loaders(batch_size=args.batch_size, workers=args.workers)

    model = resnet18_classifier(device, dataset.ds_name, pretrained=False)
    # model = WideResNet(depth=94,
    #                num_classes=10,
    #                widen_factor=16,
    #                dropRate=0.3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_model_wts = None
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        train_loss = train(model, train_loader, criterion, optimizer, transform_train, device)
        test_loss, test_acc = test(model, val_loader, criterion, norm, device)
        
        if test_acc >= best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()
            file_name = f'models/resnet18_{args.dataset}_best.pth'
            torch.save(model.state_dict(), file_name)
        
        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - start_time
        epoch_time = epoch_end_time - epoch_start_time
        estimated_time_left = epoch_time * (args.epochs - (epoch + 1))
        
        elapsed_hours, elapsed_rem = divmod(elapsed_time, 3600)
        elapsed_minutes, elapsed_seconds = divmod(elapsed_rem, 60)
        
        est_hours, est_rem = divmod(estimated_time_left, 3600)
        est_minutes, est_seconds = divmod(est_rem, 60)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}, '
              f'Elapsed Time: {int(elapsed_hours):02}:{int(elapsed_minutes):02}:{int(elapsed_seconds):02}, '
              f'Estimated Time Left: {int(est_hours):02}:{int(est_minutes):02}:{int(est_seconds):02}')

        # Update the learning rate scheduler
        scheduler.step()

    # Save the best model checkpoint
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        if not os.path.exists('models'):
            os.makedirs('models')
        file_name = f'models/resnet18_{args.dataset}.pth'
        torch.save(model.state_dict(), file_name)

        test_acc = test(model, test_loader, criterion, norm, device)[1]
        print(f'Best model saved to {file_name} with accuracy Val: {best_acc:.4f}, Test: {test_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 on MNIST or CIFAR datasets')
    parser.add_argument('--dataset', type=str, default='svhn', help='Dataset to use: mnist or cifar')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for SGD optimizer')
   
   
    args = parser.parse_args()
    main(args)
