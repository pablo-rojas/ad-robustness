import torch
import torch.nn as nn
import torchvision.models as models

from src.architectures import Patch7Descriptor, Patch17Descriptor, Patch33Descriptor, Patch65Descriptor

def main():
    # Load ResNet18 and ResNet50 (without pretrained weights for simplicity)
    resnet18 = models.resnet18(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)

    # Replace global pooling and fully-connected layers with identity to obtain feature maps
    #resnet18.avgpool = nn.Identity()
    resnet18.fc = nn.Identity()
    #resnet50.avgpool = nn.Identity()
    resnet50.fc = nn.Identity()

    # Switch models to evaluation mode to avoid BatchNorm issues with batch size 1
    resnet18.eval()
    resnet50.eval()

    print("ResNet18 feature map shapes:")
    for patch in range(5, 66):  # From 5x5 up to 33x33
        x = torch.randn(1, 3, patch, patch)
        with torch.no_grad():
            out = resnet18(x)
        print(f"Input: {patch}x{patch} -> Output shape: {out.shape}")

    print("\nResNet50 feature map shapes:")
    for patch in range(5, 66):
        x = torch.randn(1, 3, patch, patch)
        with torch.no_grad():
            out = resnet50(x)
        print(f"Input: {patch}x{patch} -> Output shape: {out.shape}")

    # Initialize and test descriptor models
    patch7 = Patch7Descriptor()
    x = torch.randn(1, 3, 7, 7)
    out = patch7(x)
    print(f"\nPatch7Descriptor output shape: {out.shape}")

    patch17 = Patch17Descriptor()
    x = torch.randn(1, 3, 17, 17)
    out = patch17(x)
    print(f"\nPatch17Descriptor output shape: {out.shape}")
    
    patch33 = Patch33Descriptor()
    x = torch.randn(1, 3, 33, 33)
    out = patch33(x)
    print(f"\nPatch33Descriptor output shape: {out.shape}")

    patch65 = Patch65Descriptor()
    x = torch.randn(1, 3, 65, 65)
    out = patch65(x)
    print(f"\nPatch65Descriptor output shape: {out.shape}")

    teacher_model = models.resnet18(pretrained=True)
    teacher_feature_extractor = torch.nn.Sequential(*list(teacher_model.children())[:-1])
    teacher_feature_extractor.eval()

    print("\nTeacher (ResNet18) feature map shapes:")
    for patch in range(5, 66):
        x = torch.randn(1, 3, patch, patch)
        with torch.no_grad():
            out = teacher_feature_extractor(x)
        print(f"Input: {patch}x{patch} -> Output shape: {out.shape}")

if __name__ == "__main__":
    main()
