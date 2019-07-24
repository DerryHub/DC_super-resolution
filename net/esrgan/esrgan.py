import torch
from torch import nn

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from esrgan_utils import RRDB, Shuffle


class Generator(nn.Module):
    def __init__(self, n=4, beta=0.1, num=8):
        super(Generator, self).__init__()
        self.firstConv = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.blocks = [RRDB(64, beta=beta) for _ in range(num)]
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
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2),  #1020
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2),  #509
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2),  #507
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 5, 2),  #252
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2),  #124
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2),  #61
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 2),  #29
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2),  #14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))

        self.linear = nn.Sequential(
            nn.Linear(128 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x1, x2):
        o1 = self.feature(x1)
        # print(o1.shape)
        o1 = o1.view([-1, 128 * 2 * 2])
        o1 = self.linear(o1)
        o2 = self.feature(x2)
        o2 = o2.view([-1, 128 * 2 * 2])
        o2 = self.linear(o2)
        o2 = torch.zeros_like(o2) + o2.mean()
        output = torch.sigmoid(o1 - o2)

        return output


if __name__ == "__main__":
    g = Generator()
    a = torch.ones([3, 3, 1024, 1024])
    # print(g(a).shape)
    d = Discriminator()
    print(d(a, a))