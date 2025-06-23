import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import (
    ResNet   as TorchResNet,
    BasicBlock as TorchBasicBlock,
    ResNet18_Weights
)


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
    def __init__(self, dim=3, padding='same'):
        super(Patch7Descriptor, self).__init__()
        # For a 7x7 patch, three conv layers with kernel size 3 reduce spatial size to 1x1.
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=3, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=padding)
        # 1x1 convolution to map 128 channels to 512.
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=padding)
        
        # Leaky ReLU with negative slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        # x: (N, dim, 7, 7)
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.decode(x)
        return x

class Patch17Descriptor(nn.Module):
    def __init__(self, dim=3, padding='same'):
        super(Patch17Descriptor, self).__init__()
        '''
        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=512, kernel_size=5, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=padding)
        
        self.decode = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=padding)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)
        '''
        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=5, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=padding)
        
        self.decode = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=padding)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

        '''
        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=128, kernel_size=5, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=padding)
        
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=padding)
        
        # Leaky ReLU with slope 5e-3
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)
        '''
        

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        
        x = self.leaky_relu(self.conv2(x))
        
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = self.decode(x)
        
        return x

class Patch33Descriptor(nn.Module):
    def __init__(self, dim=3, padding='same'):
        super(Patch33Descriptor, self).__init__()
        # same convs as before
        self.conv1    = nn.Conv2d(in_channels=dim,   out_channels=128,  kernel_size=5, stride=1, padding=padding) # Output: (B, 128, 29, 29)
        self.maxpool1 = MultiMaxPool2d(                                 kernel_size=2, stride=2)            # Output: (B, 128, 14, 14)
        self.conv2    = nn.Conv2d(in_channels=128,   out_channels=256,  kernel_size=5, stride=1, padding=padding) # Output: (B, 256, 10, 10)
        self.maxpool2 = MultiMaxPool2d(                                 kernel_size=2, stride=2)            # Output: (B, 256, 5, 5)
        self.conv3    = nn.Conv2d(in_channels=256,   out_channels=256,  kernel_size=2, stride=1, padding=padding) # Output: (B, 256, 4, 4)
        self.conv4    = nn.Conv2d(in_channels=256,   out_channels=128,  kernel_size=4, stride=1, padding=padding) # Output: (B, 128, 1, 1)
        self.decode   = nn.Conv2d(in_channels=128,   out_channels=512,  kernel_size=1, stride=1, padding=padding) # Output: (B, 512, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        B0 = x.size(0)

        x = self.leaky_relu(self.conv1(x))
        # first multipool
        x = self.maxpool1(x)                 # (B0, C1, 2,2, H1, W1)
        B1, C1, s1, s2, H1, W1 = x.shape
        x = x.view(B1 * s1 * s2, C1, H1, W1)     # merge shifts into batch

        x = self.leaky_relu(self.conv2(x))
        # second multipool
        x = self.maxpool2(x)                 # (B1*s1*s2, C2, 2,2, H2, W2)
        B2, C2, s1, s2, H2, W2 = x.shape
        x = x.view(B2 * s1 * s2, C2, H2, W2)

        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.decode(x)                      # (B2*s1*s2, 512, H3, W3)

        # reshape back & unwarp
        _, C3, H3, W3 = x.shape
        x = x.view(B2, s1, s2, C3, H3, W3)      # (B2,2,2, C3, H3, W3)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()  # (B2, C3,2,2,H3,W3)
        x = unwarp_pool(x, s1)                  # (B2, C3, H3*2, W3*2)

        _, C2, H2, W2 = x.shape
        x = x.view(B1, s1, s2, C2, H2, W2)      # (B1,2,2,C2,H2,W2)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        x = unwarp_pool(x, s1)                  # (B1, C2, H2*2, W2*2)
        return x

class Patch65Descriptor(nn.Module):
    def __init__(self, dim=3, padding='same'):
        super(Patch65Descriptor, self).__init__()
        # same convs as before
        self.conv1    = nn.Conv2d(in_channels=dim, out_channels=128,    kernel_size=5, stride=1, padding=padding) # Output: (B, 128, 64, 64)
        self.maxpool1 = MultiMaxPool2d(                                 kernel_size=2, stride=2)            # Output: (B, 128, 32, 32)
        self.conv2    = nn.Conv2d(in_channels=128, out_channels=128,    kernel_size=5, stride=1, padding=padding) # Output: (B, 128, 28, 28)
        self.maxpool2 = MultiMaxPool2d(                                 kernel_size=2, stride=2)            # Output: (B, 128, 14, 14)
        self.conv3    = nn.Conv2d(in_channels=128, out_channels=128,    kernel_size=5, stride=1, padding=padding) # Output: (B, 128, 10, 10)
        self.maxpool3 = MultiMaxPool2d(                                 kernel_size=2, stride=2)            # Output: (B, 128, 5, 5)
        self.conv4    = nn.Conv2d(in_channels=128, out_channels=256,    kernel_size=4, stride=1, padding=padding) # Output: (B, 256, 3, 3)
        self.conv5    = nn.Conv2d(in_channels=256, out_channels=128,    kernel_size=3, stride=1, padding=padding) # Output: (B, 128, 1, 1) ?????
        self.decode   = nn.Conv2d(in_channels=128, out_channels=512,    kernel_size=1, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-3)

    def forward(self, x):
        B0 = x.size(0)
        #print(f"Input shape: {x.shape}")

        x = self.leaky_relu(self.conv1(x))
        #print(f"After conv1: {x.shape}")
        x = self.maxpool1(x)
        #print(f"After maxpool1: {x.shape}")
        B1, C1, s1, s2, H1, W1 = x.shape
        x = x.view(B1 * s1 * s2, C1, H1, W1)
        #print(f"After reshaping maxpool1: {x.shape}")

        x = self.leaky_relu(self.conv2(x))
        #print(f"After conv2: {x.shape}")
        x = self.maxpool2(x)
        #print(f"After maxpool2: {x.shape}")
        B2, C2, s1, s2, H2, W2 = x.shape
        x = x.view(B2 * s1 * s2, C2, H2, W2)
        #print(f"After reshaping maxpool2: {x.shape}")

        x = self.leaky_relu(self.conv3(x))
        #print(f"After conv3: {x.shape}")
        x = self.maxpool3(x)
        #print(f"After maxpool3: {x.shape}")
        B3, C3, s1, s2, H3, W3 = x.shape
        x = x.view(B3 * s1 * s2, C3, H3, W3)
        #print(f"After reshaping maxpool3: {x.shape}")

        x = self.leaky_relu(self.conv4(x))
        #print(f"After conv4: {x.shape}")
        x = self.leaky_relu(self.conv5(x))
        #print(f"After conv5: {x.shape}")
        
        #print(f"After decode: {x.shape}")

        # reshape back & unwarp
        _, C4, H4, W4 = x.shape
        x = x.view(B3, s1, s2, C4, H4, W4)
        #print(f"After reshaping before unwarp: {x.shape}")
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        #print(f"After permute: {x.shape}")
        x = unwarp_pool(x, s1)
        #print(f"After unwarp_pool: {x.shape}")

        _, C3, H3, W3 = x.shape
        x = x.view(B2, s1, s2, C3, H3, W3)
        #print(f"After reshaping before unwarp: {x.shape}")
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        #print(f"After permute: {x.shape}")
        x = unwarp_pool(x, s1)
        #print(f"After unwarp_pool: {x.shape}")
        _, C2, H2, W2 = x.shape
        x = x.view(B1, s1, s2, C2, H2, W2)
        #print(f"After reshaping before unwarp: {x.shape}")
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        #print(f"After permute: {x.shape}")
        x = unwarp_pool(x, s1)
        #print(f"After unwarp_pool: {x.shape}")
        _, C1, H1, W1 = x.shape

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



class ResNet18ImageNet(TorchResNet):
    def __init__(
        self,
        weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
        progress: bool = True,
        **kwargs  # e.g. num_classes=1000, zero_init_residual=False, etc.
    ):
        """
        ResNet-18 backbone with ImageNet-pretrained weights + auxiliary feature methods.
        """
        # 1) Construct the *architecture* exactly as torchvision's resnet18():
        super().__init__(
            block=TorchBasicBlock,
            layers=[2, 2, 2, 2],
            **kwargs
        )

        # 2) Load the pretrained weights if requested
        if weights is not None:
            # verify and download
            weights = ResNet18_Weights.verify(weights)
            sd = weights.get_state_dict(progress=progress)
            self.load_state_dict(sd)

    def extract_features(self, x: torch.Tensor):
        feats = []
        # conv1 → bn1 → relu
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        feats.append(out)
        # maxpool
        out = self.maxpool(out)
        # every residual block
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                out = block(out)
                feats.append(out)
        # pooled vector
        pooled = F.adaptive_avg_pool2d(out, (1, 1))
        pooled = torch.flatten(pooled, 1)
        feats.append(pooled)
        return feats

    def feature_list(self, x: torch.Tensor):
        out_list = []
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.maxpool(out)
        out_list.append(out)

        out = self.layer1(out); out_list.append(out)
        out = self.layer2(out); out_list.append(out)
        out = self.layer3(out); out_list.append(out)
        out = self.layer4(out); out_list.append(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits, out_list

    def intermediate_forward(self, x: torch.Tensor, layer_index: int):
        # 0 = just after pool, 1–4 = after layer1–4
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.maxpool(out)
        if layer_index == 0:
            return out
        for idx, layer in enumerate(
            (self.layer1, self.layer2, self.layer3, self.layer4), start=1
        ):
            out = layer(out)
            if idx == layer_index:
                return out
        raise ValueError(f"layer_index must be in 0–4, got {layer_index}")

    def penultimate_forward(self, x: torch.Tensor):
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penult = self.layer4(out)
        pooled = self.avgpool(penult)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        return logits, penult

    # forward() is inherited unchanged from ResNet

# ResNet class for CIFAR-10 and MNIST
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
    
    def extract_features(self, x):
        """
        Extract features after every residual block (not just layer groups).
        This aligns with Mahalanobis-based OOD detection methodology.

        Returns:
            features: list of torch.Tensor, each being the output after a residual block
        """
        features = []

        # Initial conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        features.append(out)

        # Process through each block manually and store intermediate outputs
        for block in self.layer1:
            out = block(out)
            features.append(out)
        for block in self.layer2:
            out = block(out)
            features.append(out)
        for block in self.layer3:
            out = block(out)
            features.append(out)
        for block in self.layer4:
            out = block(out)
            features.append(out)

        # Optionally include pooled feature at the end
        pooled = F.avg_pool2d(out, 4)
        pooled = pooled.view(pooled.size(0), -1)
        features.append(pooled)

        return features
    
        # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

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
    """
    A ResNet-18 variant for dense feature extraction on CIFAR-like datasets.

    This class modifies the ResNet-18 architecture to compute dense feature maps
    by setting all strides to 1. The `self.linear` attribute is included as a placeholder
    for compatibility purposes with other ResNet implementations but is not used in the
    forward pass of this model.
    """
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

        # This is only a placeholder for compatibility purposes, but it is not used in the forward pass.
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
            self.multi_pool = MultiMaxPool2d(kernel_size=2, stride=2)
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
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.k = kernel_size
        self.s = stride

    def forward(self, x):
        B,C,H,W = x.shape
        s = self.s
        pool_outs = []
        # for each shift (i,j) only pad left/top so we stay in floor‐mode
        for i in range(s):
            for j in range(s):
                # pad only left/top, no bottom/right
                padded = F.pad(x, (j, 0, i, 0), mode='replicate')
                shifted = padded[:, :, i:(i+H), j:(j+W)]
                # floor‐mode (default)
                pooled = F.max_pool2d(shifted, kernel_size=self.k, stride=s, padding=0)
                pool_outs.append(pooled)
        # stack and reshape into (B, C, s, s, H_out, W_out)
        out = torch.stack(pool_outs, dim=2)
        B, C, SS, H_out, W_out = out.shape
        out = out.view(B, C, s, s, H_out, W_out)
        return out



if __name__ == "__main__":
    # Example usage
    model = Patch65Descriptor(dim=3)
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(output.shape) 