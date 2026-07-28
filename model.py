"""
model.py
--------
ResNet variant for the solar-cell defect classification challenge
(Deep Learning Exercise 4, FAU Erlangen-Nuernberg).

The architecture follows Table 1 of the assignment exactly:

    Conv2D(3, 64, 7, 2) -> BatchNorm -> ReLU -> MaxPool(3, 2)
    ResBlock(64,  64,  1)
    ResBlock(64,  128, 2)
    ResBlock(128, 256, 2)
    ResBlock(256, 512, 2)
    GlobalAvgPool -> Flatten -> FC(512, 2) -> Sigmoid

The final Sigmoid produces two independent probabilities (crack, inactive),
which is exactly what a multi-label problem requires.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A single residual block with two 3x3 convolutions and a skip connection.

    Structure (per the assignment):

        x -> Conv3x3(stride) -> BN -> ReLU -> Conv3x3(1) -> BN  ->(+)-> ReLU
         \\__________________ 1x1 Conv(stride) -> BN ________________/

    The 1x1 convolution on the skip path adapts the identity to the block's
    output shape, because both the number of channels and the spatial
    resolution can change (whenever ``in_channels != out_channels`` or
    ``stride != 1``).

    Parameters
    ----------
    in_channels : int
        Number of channels entering the block.
    out_channels : int
        Number of channels the block produces.
    stride : int
        Stride of the first convolution (and of the skip's 1x1 convolution).
        The second convolution always uses stride 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        # First (Conv -> BN -> ReLU). padding=1 keeps the 3x3 receptive field
        # spatially consistent; the stride here does the down-sampling.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second (Conv -> BN). Always stride 1 so it preserves resolution.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection: 1x1 conv + BN to match channels and spatial size.
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapt the identity path first, before it is overwritten.
        residual = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))  # Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))           # Conv -> BN

        out = out + residual                      # add the skip connection
        out = self.relu(out)                      # final ReLU
        return out


class ResNet(nn.Module):
    """The full ResNet as specified in Table 1 of the assignment."""

    def __init__(self) -> None:
        super().__init__()

        # Stem: aggressive 7x7 conv (stride 2) + 3x3 max-pool (stride 2).
        # 300x300 -> 150x150 (conv) -> 75x75 (pool). padding keeps sizes clean.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages. Each stride-2 block halves the spatial resolution
        # while doubling the channel count -- the classic ResNet trade-off.
        self.layer1 = ResBlock(64, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 2)
        self.layer4 = ResBlock(256, 512, 2)

        # Global average pooling collapses every 512 feature map to a single
        # value, giving a fixed-length 512-vector regardless of input size.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Classifier head: 512 -> 2, then Sigmoid for two independent
        # probabilities. Sigmoid (not Softmax) is essential for multi-label.
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
