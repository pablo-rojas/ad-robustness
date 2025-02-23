from src.architectures import ResNet18
import torchvision.models as models
from torch import nn

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model 1
model1 = models.resnet18(pretrained=False)
model1.fc = nn.Linear(model1.fc.in_features, 10)  # For CIFAR-10.
model1.to(device)

checkpoint = torch.load("models/resnet18_cifar.pth")
model1.load_state_dict(checkpoint)
model1.eval()

# Model 2
model2 = ResNet18()
model2 = torch.nn.DataParallel(model2)  # Wrap model
model2.to(device)

checkpoint = torch.load("models/ckpt.pth")
model2.load_state_dict(checkpoint['net'])
model2.eval()

print ("Model comparisson")
print ()
print ("Model1")
print (model1)
print ()
print ("Model2")
print (model2)