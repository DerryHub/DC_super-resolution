import os
from estimate import PSNR, SSIM
from net.carn.carn import CARN
from net.carn.carn_m import CARN_M
from net.edsr.edsr import EDSR
import net.srgan.SRGAN as SR
import net.esrgan.esrgan as ESR
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import time
import numpy as np
from dataset import TestDataset

root = os.path.dirname(__file__)

model = 'CARN'
n = 4
useCUDA = True

dataset = TestDataset(n=n)

topil = transforms.ToPILImage()

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

if useCUDA:
    net = net.cuda()

net.load_state_dict(
    torch.load(
        os.path.join(root, 'models/{}_{}.pkl'.format(model, n)),
        map_location={'cuda:1': 'cuda:0'}))

if not os.path.exists(os.path.join(root, 'DC_data/test-images_original/')):
    os.makedirs(os.path.join(root, 'DC_data/test-images_original/'))

for i, xnImg in enumerate(tqdm(dataset)):
    xnImg = torch.unsqueeze(xnImg, 0)
    if useCUDA:
        xnImg = xnImg.cuda()
    with torch.no_grad():
        if useCUDA:
            xnImg = xnImg.cuda()
            originalImage = net(xnImg).cpu()[0]
        else:
            originalImage = net(xnImg)[0]
    originalImage = torch.clamp(originalImage, 0, 1)
    originalImage = topil(originalImage)
    originalImage.save(os.path.join(root, 'DC_data/test-images_original/', 'test_original ({}).png'.format(i+1)))
