import torch
import os
from net.carn.carn import CARN
from PIL import Image
from torchvision import transforms

model = 'CARN'

root = os.path.dirname(__file__)

if model == 'CARN':
    net = CARN()

net.load_state_dict(torch.load(os.path.join(root, 'models/{}.pkl'.format(model))))

x4Img = Image.open(
    os.path.join(root, 'DC_data/val-images/val-images_x4/val_x4 (1).png'))

originImg = Image.open(
    os.path.join(
        root, 'DC_data/val-images/val-images_original/val_original (1).png'))

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

preImg = net(torch.unsqueeze(totensor(x4Img), 0))

preImg = topil(preImg[0])

originImg.show()
x4Img.show()
preImg.show()
