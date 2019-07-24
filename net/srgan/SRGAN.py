import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

root = os.path.dirname(__file__)
sys.path.append(root)

from srgan_utils import *

class Generator(nn.Module):
    def __init__(self,
                 n_residual_blocks,
                 upsample_factor,
                 num_channel=3,
                 base_filter=64):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.conv1 = nn.Conv2d(
            num_channel, base_filter, kernel_size=9, stride=1, padding=4)
        for i in range(self.n_residual_blocks):
            self.add_module(
                'residual_block' + str(i + 1),
                ResidualBlock(
                    in_channels=base_filter,
                    out_channels=base_filter,
                    kernel=3,
                    stride=1))
        self.conv2 = nn.Conv2d(
            base_filter, base_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)
        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1),
                            UpsampleBlock(base_filter))
        self.conv3 = nn.Conv2d(
            base_filter, num_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
        x = self.bn2(self.conv2(y)) + x
        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)
        out = self.conv3(x)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, num_channel=3, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channel, base_filter, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            DiscriminatorBlock(
                base_filter, base_filter, kernel_size=3, stride=2, padding=1),
            DiscriminatorBlock(
                base_filter,
                base_filter * 2,
                kernel_size=3,
                stride=1,
                padding=1),
            DiscriminatorBlock(
                base_filter * 2,
                base_filter * 2,
                kernel_size=3,
                stride=2,
                padding=1),
            DiscriminatorBlock(
                base_filter * 2,
                base_filter * 4,
                kernel_size=3,
                stride=1,
                padding=1),
            DiscriminatorBlock(
                base_filter * 4,
                base_filter * 4,
                kernel_size=3,
                stride=2,
                padding=1),
            DiscriminatorBlock(
                base_filter * 4,
                base_filter * 8,
                kernel_size=3,
                stride=1,
                padding=1),
            DiscriminatorBlock(
                base_filter * 8,
                base_filter * 8,
                kernel_size=3,
                stride=2,
                padding=1),
        )
        self.conv2 = nn.Conv2d(
            base_filter * 8, 1, kernel_size=1, stride=1)

    def forward(self, x):
        out = swish(self.conv1(x))
        out = self.blocks(out)
        out = self.conv2(out)
        out = torch.sigmoid(F.avg_pool2d(out,
                                         out.size()[2:])).view(
                                             out.size()[0], -1)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


if __name__ == "__main__":
    Gnet = Generator(16, 4).cuda()
    Dnet = Discriminator().cuda()
    input = torch.ones([1, 3, 100, 100]).cuda()
    Goutput = Gnet(input)
    Doutput = Dnet(Goutput)
    print('Goutput', Goutput.shape)
    print('Doutput', Doutput.shape)
