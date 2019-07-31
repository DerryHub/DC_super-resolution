from torch import nn
import torch
import os

root = os.path.dirname(__file__)

class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1), nn.LeakyReLU())
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, padding=1),
            nn.LeakyReLU())
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, padding=1),
            nn.LeakyReLU())
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 3, 1, padding=1),
            nn.LeakyReLU())
        self.lastConv = nn.Conv2d(
            in_channels * 5, in_channels, 3, 1, padding=1)

    def forward(self, x):
        c1_out = self.Conv1(x)
        c2_in = torch.cat([x, c1_out], dim=1)

        c2_out = self.Conv2(c2_in)
        c3_in = torch.cat([c2_in, c2_out], dim=1)

        c3_out = self.Conv3(c3_in)
        c4_in = torch.cat([c3_in, c3_out], dim=1)

        c4_out = self.Conv4(c4_in)
        output = torch.cat([c4_in, c4_out], dim=1)

        output = self.lastConv(output)

        return output

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, beta=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.beta = beta
        self.db1 = DenseBlock(in_channels)
        self.db2 = DenseBlock(in_channels)
        self.db3 = DenseBlock(in_channels)

    def forward(self, x):
        o1 = x + self.beta * self.db1(x)
        o2 = o1 + self.beta * self.db2(o1)
        o3 = o2 + self.beta * self.db3(o2)

        output = x + self.beta * o3

        return output

class Shuffle(nn.Module):
    def __init__(self, in_channels, n=4):
        super(Shuffle, self).__init__()
        if n == 4:
            self.shuffle = nn.Sequential(
                nn.Conv2d(in_channels, 4 * in_channels, 3, 1, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels, 4 * in_channels, 3, 1, padding=1),
                nn.PixelShuffle(2))
        elif n == 2:
            self.shuffle = nn.Sequential(
                nn.Conv2d(in_channels, 4 * in_channels, 3, 1, padding=1),
                nn.PixelShuffle(2))

    def forward(self, x):
        output = self.shuffle(x)
        return output