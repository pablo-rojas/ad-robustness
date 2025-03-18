import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn




def unwarp_pool(x, s):
    """
    Unwarps the multipooling output.
    Input shape: (B, C, s, s, H, W)
    Output shape: (B, C, H * s, W * s)
    """
    B, C, s1, s2, H, W = x.shape
    # Permute to interleave the offset dimensions with the spatial dims:
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # (B, C, H, s, W, s)
    x = x.view(B, C, H * s1, W * s2)
    return x

def modify_resnet_for_dense(model):
    """
    Modify a ResNet model to have a stride of 1 for dense prediction tasks.

    This function adjusts the stride of the initial convolutional layer, the maxpool layer (if it exists),
    and the first convolutional layer in each block of layers 2, 3, and 4 to (1, 1). Additionally, it modifies
    the stride of the shortcut (downsampling) path in these blocks if they exist.

    Args:
        model (torch.nn.Module): The ResNet model to be modified.

    Returns:
        torch.nn.Module: The modified ResNet model with updated strides.
    """
    model.conv1.stride = (1, 1)
    if hasattr(model, 'maxpool'):
        model.maxpool.stride = (1, 1)
    for layer in [model.layer2, model.layer3, model.layer4]:
        # Set the first convolution's stride to 1 in each block.
        layer[0].conv1.stride = (1, 1)
        # Check if the block has a shortcut (downsampling) path and modify its stride.
        if hasattr(layer[0], 'shortcut') and len(layer[0].shortcut) > 0:
            # Assuming the first layer in the shortcut is a Conv2d, set its stride to 1.
            if isinstance(layer[0].shortcut[0], nn.Conv2d):
                layer[0].shortcut[0].stride = (1, 1)
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            layer[0].downsample[0].stride = (1, 1)
    return model

class Patch7Descriptor(nn.Module):
    def __init__(self, dim=3):
        super(Patch7Descriptor, self).__init__()
        # For a 7x7 patch, three conv layers with kernel size 3 reduce spatial size to 1x1.
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 1x1 convolution to map 256 channels to 512.
        self.decode = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        
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
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        
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

class DenseCIFARResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dim=3):
        super(DenseCIFARResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(dim, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)

        # This is only a placeholder for compatibility pruposes, but it is not used in the forward pass.
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
        return out

class DenseResNet18(nn.Module):
    """
    Wraps a torchvision ResNet-18 to compute dense feature maps in one forward pass.
    In this example, we convert the first maxpool layer to a multipooling module.
    (A full conversion would similarly handle strided convolutions in later layers.)
    """
    def __init__(self, orig_resnet):
        super(DenseResNet18, self).__init__()
        # Use the early layers from resnet18
        self.conv1 = orig_resnet.conv1   # (kernel=7, stride=2, padding=3)
        self.bn1 = orig_resnet.bn1
        if hasattr(orig_resnet, 'relu'):
            self.relu = orig_resnet.relu
        else:
            self.relu = F.relu
        # Replace the first maxpool with our multipooling.
        # Original maxpool: kernel_size=3, stride=2, padding=1.
        # (Here we assume no extra padding; in practice you might need to pad/crop so that each pixel’s patch is processed.)
        if hasattr(orig_resnet, 'maxpool'):
            self.multi_pool = MultiMaxPool2d(kernel_size=3, stride=2)
            self.cifar = False
        # Keep the remaining layers unchanged (they are fully convolutional).
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        # For a “feature extractor” we drop the average pool and fc layer.

    def forward(self, x):
        # x: input image of shape (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar:
            x = self.multi_pool(x)
            B, C, s1, s2, H1, W1 = x.shape
            x = x.view(B * s1 * s2, C, H1, W1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if not self.cifar:
            # Let the new spatial size be H2 x W2 and output channels C_out.
            B_new, C_out, H2, W2 = x.shape
            # Reshape back: split batch back into (B, s1, s2)
            x = x.view(B, s1, s2, C_out, H2, W2)
            # Rearrange dimensions to (B, C_out, s1, s2, H2, W2)
            x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
            # Finally, unwarp the multipool dimensions to get a dense output.
            x = unwarp_pool(x, s1)  # (B, C_out, H2 * s, W2 * s)
        return x
    

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

class MultiMaxPool2d(nn.Module):
    """
    A multipooling layer that computes s×s shifted max-pooling outputs.
    For each offset (i, j) in {0, …, s-1}^2, it applies maxpooling on a shifted input.
    """
    def __init__(self, kernel_size, stride):
        super(MultiMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        s = self.stride
        pool_outs = []
        # Compute pooling for every shift (i,j)
        for i in range(s):
            for j in range(s):
                # Shift the input by slicing (assumes proper padding beforehand)
                shifted = x[:, :, i:, j:]
                pooled = F.max_pool2d(shifted, kernel_size=self.kernel_size, stride=self.stride, padding=1)
                pool_outs.append(pooled)
        # Stack and reshape: initial shape (B, C, s*s, H_out, W_out)
        out = torch.stack(pool_outs, dim=2)
        # Reshape to (B, C, s, s, H_out, W_out)
        out = out.view(x.size(0), x.size(1), s, s, out.size(-2), out.size(-1))
        return out

