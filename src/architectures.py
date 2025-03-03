import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class Patch7Descriptor(nn.Module):
    def __init__(self, dim=3):
        super(Patch7Descriptor, self).__init__()
        # For a 7x7 patch, three conv layers with kernel size 3 reduce spatial size to 1x1.
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        # 1x1 convolution to map 256 channels to 512.
        self.decode = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
        
        # Leaky ReLU with negative slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        # x: (N, dim, 7, 7)
        x = self.leaky_relu(self.conv1(x))  # (N, 128, 5, 5)
        x = self.leaky_relu(self.conv2(x))  # (N, 256, 3, 3)
        x = self.leaky_relu(self.conv3(x))  # (N, 256, 1, 1)
        x = self.decode(x)                  # (N, 512, 1, 1)
        return x

class Patch17Descriptor(nn.Module):
    def __init__(self, dim=3):
        super(Patch17Descriptor, self).__init__()

        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
        
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        
        x = self.leaky_relu(self.conv2(x))
        
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = self.decode(x)
        
        return x

class Patch33Descriptor(nn.Module):
    def __init__(self, dim=3):
        super(Patch33Descriptor, self).__init__()

        # Architecture for p = 33
        self.conv1      = nn.Conv2d(in_channels=dim, out_channels=128,  kernel_size=5, stride=1)
        self.maxpool1   = nn.MaxPool2d(                                 kernel_size=2, stride=2)
        self.conv2      = nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=5, stride=1)
        self.maxpool2   = nn.MaxPool2d(                                 kernel_size=2, stride=2)
        self.conv3      = nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=2, stride=1)
        self.conv4      = nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=4, stride=1)
        self.decode     = nn.Conv2d(in_channels=128, out_channels=512,  kernel_size=1, stride=1)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.leaky_relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = self.decode(x)
        
        return x

class Patch65Descriptor(nn.Module):
    def __init__(self, dim=3):
        super(Patch65Descriptor, self).__init__()
        
        # Define layers as per the table
        self.conv1      = nn.Conv2d(in_channels=dim, out_channels=128,  kernel_size=5, stride=1)
        self.maxpool1   = nn.MaxPool2d(                                 kernel_size=2, stride=2)
        self.conv2      = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=5, stride=1)
        self.maxpool2   = nn.MaxPool2d(                                 kernel_size=2, stride=2)
        self.conv3      = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=5, stride=1)
        self.maxpool3   = nn.MaxPool2d(                                 kernel_size=2, stride=2)
        self.conv4      = nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=4, stride=1)
        self.conv5      = nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=3, stride=1)
        self.decode     = nn.Conv2d(in_channels=128, out_channels=512,  kernel_size=1, stride=1)

        # Activation function (Leaky ReLU with slope 5e-3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.leaky_relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.leaky_relu(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        
        x = self.decode(x)
        
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dim=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(dim, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(dim=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], dim=dim)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
