import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels, out_channels, stride=stride, dilation=dilation
        )
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, dilation=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, reduction=4):
        super().__init__()
        # Reduce channels by reduction factor
        bottleneck_channels = in_channels // reduction

        # Bottleneck path
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class MushroomClassifier(nn.Module):
    """
    Defines a CNN used to classify mushrooms
    The forward pass flattens the max and average pool layers and concatenates
        them before feeding them into the fully connected layer
    """

    def __init__(self, num_classes):
        super().__init__()

        self.initial = nn.Sequential(
            ConvBlock(3, 16, stride=1),
            ConvBlock(16, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block1 = nn.Sequential(
            BottleneckResBlock(32, 64), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            BottleneckResBlock(64, 128), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.block3 = self.block2 = nn.Sequential(
            BottleneckResBlock(128, 256, dilation=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout2 = nn.Dropout(0.2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * 2, num_classes)
        n_parameters = sum(p.numel() for p in self.parameters())
        print(f"Created a model with {n_parameters} parameters")

    def forward(self, x):
        out = self.initial(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.dropout1(out)
        out = self.block3(out)
        out = self.dropout2(out)
        max_pooled = self.max_pool(out)
        avg_pooled = self.avg_pool(out)
        max_pooled = torch.flatten(max_pooled, 1)
        avg_pooled = torch.flatten(avg_pooled, 1)
        out = torch.cat([max_pooled, avg_pooled], dim=1)
        out = self.dropout3(out)
        out = self.fc(out)

        return out
