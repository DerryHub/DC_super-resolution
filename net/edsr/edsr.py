import torch
from torch import nn

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from edsr_utils import ResidualBlock, Shuffle


class EDSR(nn.Module):
    def __init__(self, num=32, n=4):
        super(EDSR, self).__init__()

        self.firstConv = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.lastConv = nn.Conv2d(64, 3, 3, 1, padding=1)

        self.ResidualBlocks = [ResidualBlock(64) for _ in range(num)]
        self.ResidualBlocks.append(nn.Conv2d(64, 64, 3, 1, padding=1))

        self.RBs = nn.Sequential(*self.ResidualBlocks)

        self.shuffle = Shuffle(64, n=n)

    def forward(self, x):
        output = self.firstConv(x)
        output = self.RBs(output)
        output = self.shuffle(output)
        output = self.lastConv(output)
        return output


if __name__ == "__main__":
    edsr = EDSR()
    a = torch.ones([1, 3, 100, 100])
    print(edsr(a).shape)