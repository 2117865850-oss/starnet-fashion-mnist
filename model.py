"""AI-generated small Fashion-MNIST adaptation of Ma et al.'s StarNet.

Block topology based on the Apache-2.0 official implementation:
https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py
This is a small adaptation, not one of the published ImageNet configurations.
"""
import torch
from torch import nn


def conv_bn(cin, cout, kernel, stride=1, groups=1, bn=True):
    layers = [nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                        groups=groups, bias=True)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    return nn.Sequential(*layers)


class StarBlock(nn.Module):
    def __init__(self, channels, operation="star", expansion=2):
        super().__init__()
        if operation not in {"star", "sum"}:
            raise ValueError("operation must be 'star' or 'sum'")
        self.operation = operation
        self.dw1 = conv_bn(channels, channels, 7, groups=channels)
        self.branch1 = conv_bn(channels, expansion * channels, 1, bn=False)
        self.branch2 = conv_bn(channels, expansion * channels, 1, bn=False)
        self.project = conv_bn(expansion * channels, channels, 1)
        self.dw2 = conv_bn(channels, channels, 7, groups=channels, bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        z = self.dw1(x)
        a, b = self.act(self.branch1(z)), self.branch2(z)
        z = a * b if self.operation == "star" else a + b
        return x + self.dw2(self.project(z))


class FashionStarNet(nn.Module):
    def __init__(self, operation="star"):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, 16, 3), nn.ReLU6())
        self.stage1 = StarBlock(16, operation)
        self.stage2 = nn.Sequential(conv_bn(16, 32, 3, stride=2),
                                    StarBlock(32, operation))
        self.stage3 = nn.Sequential(conv_bn(32, 64, 3, stride=2),
                                    StarBlock(64, operation))
        self.norm = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)
        # Deliberately use PyTorch's default Conv2d/Linear initialization.
        # Paired star/sum models load the exact same initial state dictionary.

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(self.pool(self.norm(x)).flatten(1))


if __name__ == "__main__":
    model = FashionStarNet()
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Output shape:", tuple(model(torch.zeros(2, 1, 28, 28)).shape))
