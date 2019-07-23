import torch
from torch import nn
import torch.nn.functional as F




class ResidualEBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(ResidualEBlock, self).__init__()

        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, 1, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, 3, 1, padding=1, groups=groups),
            nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, 1, 1))

    def forward(self, x):
        branchOut = self.branch(x)
        output = x + branchOut
        output = F.relu(output, inplace=True)
        return output


class CascadingBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(CascadingBlock, self).__init__()

        self.residual_1 = ResidualEBlock(in_channels, groups)
        self.residual_2 = ResidualEBlock(in_channels, groups)
        self.residual_3 = ResidualEBlock(in_channels, groups)

        self._1x1Conv_1 = nn.Conv2d(2 * in_channels, in_channels, 1, 1)
        self._1x1Conv_2 = nn.Conv2d(3 * in_channels, in_channels, 1, 1)
        self._1x1Conv_3 = nn.Conv2d(4 * in_channels, in_channels, 1, 1)

    def forward(self, x):
        r1_out = self.residual_1(x)
        c1_in = torch.cat([x, r1_out], dim=1)
        c1_out = self._1x1Conv_1(c1_in)

        r2_out = self.residual_2(c1_out)
        c2_in = torch.cat([x, r1_out, r2_out], dim=1)
        c2_out = self._1x1Conv_2(c2_in)

        r3_out = self.residual_3(c2_out)
        c3_in = torch.cat([x, r1_out, r2_out, r3_out], dim=1)
        c3_out = self._1x1Conv_3(c3_in)

        return c3_out


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
    # c = CascadingBlock(64).cuda()
    # input = torch.ones([1, 64, 300, 300]).cuda()
    # output = c(input)
    # print(output.shape)
    import os
    import sys
    root = os.path.dirname(__file__)
    sys.path.append(os.path.join(root, '../..'))
    from dataset import MyDataset
    dataset = MyDataset()

