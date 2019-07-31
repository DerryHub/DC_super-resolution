import torch
import os
from net.carn.carn import CARN
from net.carn.carn_m import CARN_M
from net.edsr.edsr import EDSR
from net.esrg.esrg import ESRG
import net.srgan.SRGAN as SR
import net.esrgan.esrgan as ESR
from PIL import Image
from torchvision import transforms

model = 'EDSR'
n = 4

root = os.path.dirname(__file__)

if model == 'CARN':
    net = CARN(n=n)
elif model == 'CARN_M':
    net = CARN_M(n=n)
elif model == 'EDSR':
    net = EDSR(n=n)
elif model == 'SRGAN':
    net = SR.Generator(n_residual_blocks=8, upsample_factor=n)
elif model == 'ESRGAN':
    net = ESR.Generator(n=n, num=4)
elif model == 'ESRG':
    net = ESRG(n=n, num=4)

net.load_state_dict(
    torch.load(
        os.path.join(root, 'models/{}_{}.pkl'.format(model, n)),
        map_location={'cuda:1': 'cuda:0'}))

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

with torch.no_grad():
    preImg = net(torch.unsqueeze(totensor(xnImg), 0))
preImg = torch.clamp(preImg, 0, 1)
preImg = topil(preImg[0])

originImg.show()
xnImg.show()
preImg.show()
