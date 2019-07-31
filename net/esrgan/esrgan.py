import torch
from torch import nn

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from esrgan_utils import ResidualInResidualDenseBlock, FeatureExtractor, Shuffle


class Generator(nn.Module):
    def __init__(self, n=4, beta=0.2, num=16):
        super(Generator, self).__init__()
        self.firstConv = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.blocks = [
            ResidualInResidualDenseBlock(in_channels=64, beta=beta)
            for _ in range(num)
        ]
        self.blocks.append(nn.Conv2d(64, 64, 3, 1, padding=1))
        self.BBs = nn.Sequential(*self.blocks)
        self.shuffle = Shuffle(64, n=n)
        self.lastConv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.Conv2d(64, 3, 3, 1, padding=1))

    def forward(self, x):
        o1 = self.firstConv(x)
        o2 = o1 + self.BBs(o1)
        output = self.shuffle(o2)
        output = self.lastConv(output)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        in_filters = 3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

if __name__ == "__main__":
    g = Discriminator().cuda()
    a = torch.ones([1, 3, 1024, 1024]).cuda()
    print(g(a).shape)