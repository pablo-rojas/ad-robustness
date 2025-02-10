from torch import nn

class ShallowNet(nn.Module):
    def __init__(self, num_channels=3):
        super(ShallowNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
class Patch17Descriptor(nn.Module):
    def __init__(self):
        super(Patch17Descriptor, self).__init__()

        # Architecture for p = 17
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1)
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
    def __init__(self):
        super(Patch33Descriptor, self).__init__()

        # Architecture for p = 33
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1)
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)
        
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
    def __init__(self):
        super(Patch65Descriptor, self).__init__()
        
        # Define layers as per the table
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decode = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1)

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
