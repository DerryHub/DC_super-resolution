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

root = os.path.dirname(__file__)

model = 'EDSR'
n = 4
useCUDA = True

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

originPath = 'DC_data/val-images/val-images_original/'
xnPath = 'DC_data/val-images/val-images_x{}/'.format(n)

PSNR_list = []
SSIM_list = []

t = 0

for i in tqdm(range(100)):
    originName = 'val_original ({}).png'.format(i + 1)
    xnName = 'val_x{} ({}).png'.format(n, i + 1)

    originImage = Image.open(os.path.join(root, originPath, originName))
    xnImage = Image.open(os.path.join(root, xnPath, xnName))

    totensor = transforms.ToTensor()

    t0 = time.clock()
    with torch.no_grad():
        if useCUDA:
            preImg = net(torch.unsqueeze(totensor(xnImage),
                                         0).cuda())[0].cpu().detach().numpy()
        else:
            preImg = net(torch.unsqueeze(totensor(xnImage),
                                         0))[0].detach().numpy()
    preImg = np.clip(preImg, 0, 1)
    t1 = time.clock()
    originImage = totensor(originImage).detach().numpy()

    psnr = PSNR(originImage, preImg)
    ssim = SSIM(originImage, preImg)

    PSNR_list.append(psnr)
    SSIM_list.append(ssim)

    t += t1 - t0

print('mean PSNR is {}'.format(sum(PSNR_list) / len(PSNR_list)))
print('mean SSIM is {}'.format(sum(SSIM_list) / len(SSIM_list)))
print('time is {}'.format(t))