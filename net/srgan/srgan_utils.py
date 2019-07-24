import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x*torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,kernel,out_channels,stride):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride = stride,padding = kernel//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel,stride=stride,padding=kernel//2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out1 = self.bn1(self.conv2(x))
        y = swish(out1)
        out = self.bn2(self.conv2(y)) + x
        return out

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels):
        super(UpsampleBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels*4,kernel_size=3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(2)
    def forward(self,x):
        out = self.conv(x)
        out = self.shuffler(out)
        out = swish(out)
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(DiscriminatorBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = swish(out)
        return out
