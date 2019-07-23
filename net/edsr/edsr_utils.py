import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1),
        )

    def forward(self, x):
        b = self.branch(x)
        output = x + b*0.1
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

if __name__ == "__main__":
    rb = ResidualBlock(100)
    a = torch.ones([10, 100, 200, 200])
    print(rb(a).shape)