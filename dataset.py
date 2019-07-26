import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

root = os.path.dirname(__file__)


class MyDataset(Dataset):
    def __init__(self, train=True, n=4):
        self.n = n
        self.train = train

        if train:
            self.originPath = 'DC_data/trainData/'
        else:
            self.originPath = 'DC_data/val-images/val-images_original/'

        self.xnPath = 'DC_data/val-images/val-images_x{}/'.format(n)

        self.fileList = os.listdir(os.path.join(root, self.originPath))[:100]
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index):
        filename = self.fileList[index]
        if self.train:
            originImage = Image.open(
                os.path.join(root, self.originPath, filename))
            w, h = originImage.size
            xnImage = originImage.resize((w // self.n, h // self.n),
                                         Image.BICUBIC)
        else:
            xnImage = Image.open(
                os.path.join(root, self.xnPath, 'val_x{} ({}).png'.format(
                    self.n, index + 1)))
            originImage = Image.open(
                os.path.join(root, self.originPath,
                             'val_original ({}).png'.format(index + 1)))

        originImage = self.totensor(originImage)
        xnImage = self.totensor(xnImage)
        return originImage, xnImage


if __name__ == "__main__":
    mydataset = MyDataset()
    print(len(mydataset))
    for o, x in mydataset:
        print(o.shape, x.shape)