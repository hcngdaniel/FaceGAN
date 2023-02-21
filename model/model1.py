#!/usr/bin/env python3
import torch
from torch import nn


class ConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLUBlock, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.Conv(x))


class ConvGELUBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvGELUBNBlock, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.GELU(self.BN(self.Conv(x)))


class TransConvGELUBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransConvGELUBNBlock, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.GELU(self.BN(self.Conv(x)))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5 * 50 * 25),
        )
        self.unflatten = nn.Unflatten(1, (5, 50, 25))
        self.transconv = nn.Sequential(
            TransConvGELUBNBlock(5, 5, 1, 1, 0),
            TransConvGELUBNBlock(5, 5, 1, 1, 0),
            TransConvGELUBNBlock(5, 10, 1, 1, 0),
            TransConvGELUBNBlock(10, 10, 1, 1, 0),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.Upsample((100, 50)),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.Upsample((150, 100)),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.Upsample((200, 150)),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.Upsample((200, 150)),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 10, 5, 1, 2),
            TransConvGELUBNBlock(10, 10, 3, 1, 1),
            TransConvGELUBNBlock(10, 3, 5, 1, 2),
            nn.Upsample((218, 178)),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 5, 1, 2),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 5, 1, 2),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 5, 1, 2),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 3, 1, 1),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            TransConvGELUBNBlock(3, 3, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.transconv(self.unflatten(self.linear(x)))


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.backbone = nn.Sequential(
            ConvReLUBlock(3, 5, 3, 1, 1),
            ConvReLUBlock(5, 5, 5, 1, 2),
            ConvReLUBlock(5, 10, 3, 1, 1),
            ConvReLUBlock(10, 10, 5, 1, 2),
            nn.AdaptiveAvgPool2d((200, 150)),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.AdaptiveAvgPool2d((150, 100)),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.AdaptiveAvgPool2d((100, 50)),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            ConvGELUBNBlock(10, 10, 3, 1, 1),
            ConvGELUBNBlock(10, 10, 5, 1, 2),
            nn.AdaptiveAvgPool2d((50, 25)),
            ConvGELUBNBlock(10, 5, 3, 1, 1),
            ConvGELUBNBlock(5, 5, 5, 1, 2),
            ConvGELUBNBlock(5, 5, 3, 1, 1),
            ConvGELUBNBlock(5, 5, 5, 1, 2),
            ConvGELUBNBlock(5, 5, 3, 1, 1),
            ConvGELUBNBlock(5, 5, 5, 1, 2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(5 * 50 * 25, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.linear(self.flatten(self.backbone(x)))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.generator = Generator()
        self.critic = Critic()
