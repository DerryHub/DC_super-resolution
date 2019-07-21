import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

root = os.path.dirname(__file__)


class MyDataset(Dataset):
    def __init__(self, train=True):
        originPath = 'DC_data/DIV2K-dataset/DIV2K_{}_HR'
        X4Path = 'DC_data/DIV2K-dataset/DIV2K_{}_LR_bicubic_X4/DIV2K_{}_LR_bicubic/X4'
        if train:
            originPath = originPath.format('train')
            X4Path = X4Path.format('train', 'train')
        else:
            originPath = originPath.format('valid')
            X4Path = X4Path.format('valid', 'valid')

        self.originPath = os.path.join(root, originPath)
        self.X4Path = os.path.join(root, X4Path)

        self.fileList = os.listdir(originPath)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index):
        filename = self.fileList[index]
        originImage = Image.open(os.path.join(self.originPath, filename))
        originImage = originImage.resize((756, 564), Image.BICUBIC)
        # print(originImage.size)
        # x4Image = Image.open(
        #     os.path.join(self.X4Path,
        #                  filename.split('.')[0] + 'x4.png'))
        x4Image = originImage.resize((189, 141), Image.BICUBIC)
        originImage = self.totensor(originImage)
        x4Image = self.totensor(x4Image)
        return originImage, x4Image


if __name__ == "__main__":
    mydataset = MyDataset()
    print(mydataset[0][1])