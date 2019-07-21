import torch
from torch import nn

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from utils import CascadingBlock, Shuffle

class CARN(nn.Module):
    def __init__(self, groups=1):
        super(CARN, self).__init__()

        self.firstConv = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.lastConv = nn.Conv2d(64, 3, 3, 1, padding=1)

        self.cascading_1 = CascadingBlock(64, groups=groups)
        self.cascading_2 = CascadingBlock(64, groups=groups)
        self.cascading_3 = CascadingBlock(64, groups=groups)

        self._1x1Conv_1 = nn.Conv2d(64 * 2, 64, 1, 1)
        self._1x1Conv_2 = nn.Conv2d(64 * 3, 64, 1, 1)
        self._1x1Conv_3 = nn.Conv2d(64 * 4, 64, 1, 1)

        self.shuffle = Shuffle(64)

    def forward(self, x):
        x = 2*x - 1

        firstConv_out = self.firstConv(x)

        ca1_out = self.cascading_1(firstConv_out)
        c1_in = torch.cat([firstConv_out, ca1_out], dim=1)
        c1_out = self._1x1Conv_1(c1_in)

        ca2_out = self.cascading_2(c1_out)
        c2_in = torch.cat([firstConv_out, ca1_out, ca2_out], dim=1)
        c2_out = self._1x1Conv_2(c2_in)

        ca3_out = self.cascading_3(c2_out)
        c3_in = torch.cat([firstConv_out, ca1_out, ca2_out, ca3_out], dim=1)
        c3_out = self._1x1Conv_3(c3_in)

        shuffle_out = self.shuffle(c3_out)

        output = self.lastConv(shuffle_out)

        output = (output+1)/2

        return output


if __name__ == "__main__":
    carn = CARN().cuda()
    input = torch.ones([1, 3, 100, 100]).cuda()
    output = carn(input)
    print(output.shape)