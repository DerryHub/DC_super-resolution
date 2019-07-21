import torch
import os
from net.carn.carn import CARN
from net.carn.carn_m import CARN_M
from PIL import Image
from torchvision import transforms

model = 'CARN_M'
n = 2

root = os.path.dirname(__file__)

if model == 'CARN':
    net = CARN(n=n)
elif model == 'CARN_M':
    net = CARN_M(n=n)

net.load_state_dict(
    torch.load(os.path.join(root, 'models/{}_{}.pkl'.format(model, n))))

if n == 4:
    xnImg = Image.open(
        os.path.join(root, 'DC_data/val-images/val-images_x4/val_x4 (1).png'))
elif n == 2:
    xnImg = Image.open(
        os.path.join(root, 'DC_data/val-images/val-images_x2/val_x2 (1).png'))

originImg = Image.open(
    os.path.join(
        root, 'DC_data/val-images/val-images_original/val_original (1).png'))

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

preImg = net(torch.unsqueeze(totensor(xnImg), 0))

preImg = topil(preImg[0])

originImg.show()
xnImg.show()
preImg.show()
