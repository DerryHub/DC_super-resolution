import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import net.srgan.SRGAN as SR
import net.esrgan.esrgan as ESR

root = os.path.dirname(__file__)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)

# useCUDA = True
model = 'SRGAN'
lr = 1e-3
EPOCH = 20
batch_size = 1
loadModel = False
n = 4

trainLoader = DataLoader(
    MyDataset(train=True, n=n),
    batch_size=batch_size,
    shuffle=True,
    num_workers=6)
validLoader = DataLoader(
    MyDataset(train=False, n=n),
    batch_size=batch_size,
    shuffle=False,
    num_workers=6)

if model == 'SRGAN':
    generaotr = SR.Generator(n_residual_blocks=8, upsample_factor=n).to(device)
    discriminator = SR.Discriminator().to(device)
    generaotr.weight_init(mean=0.0, std=0.2)
    discriminator.weight_init(mean=0.0, std=0.2)
elif model == 'ESRGAN':
    generaotr = ESR.Generator(n=n, num=4).to(device)
    discriminator = ESR.Discriminator().to(device)

print('Using {}'.format(model))

if not os.path.exists(os.path.join(root, 'models/')):
    os.mkdir(os.path.join(root, 'models/'))

if loadModel and os.path.exists(
        os.path.join(root, 'models/{}_{}.pkl'.format(model, n))):
    print('Loading {} model......'.format(model))
    generaotr.load_state_dict(
        torch.load(os.path.join(root, 'models/{}_{}.pkl'.format(model, n))))
    discriminator.load_state_dict(
        torch.load(
            os.path.join(
                root, 'models/{}_{}.pkl'.format(model + '_discriminator', n))))

# costG = nn.MSELoss()
costG = nn.L1Loss()
costD = nn.BCELoss()

optG = optim.Adam(generaotr.parameters(), lr=lr, betas=(0.9, 0.999))
optD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
steplrG = torch.optim.lr_scheduler.StepLR(optG, 5)
steplrD = torch.optim.lr_scheduler.StepLR(optD, 10)

MinValidLoss = np.inf
MinGLoss = np.inf
MinDLoss = np.inf

for epoch in range(EPOCH):
    steplrG.step()
    steplrD.step()
    trainLoader_t = tqdm(trainLoader)
    trainLoader_t.set_description_str(' Train epoch {}'.format(epoch))
    GlossList = []
    DlossList = []
    generaotr.train()
    discriminator.train()
    for originImage, xnImage in trainLoader_t:
        originImage = originImage.to(device)
        xnImage = xnImage.to(device)

        real_label = torch.ones(xnImage.size(0), 1).to(device)
        fake_label = torch.zeros(xnImage.size(0), 1).to(device)

        #训练Ｄ网络
        optD.zero_grad()
        preImg = generaotr(xnImage)
        if model == 'SRGAN':
            d_real = discriminator(originImage)
        elif model == 'ESRGAN':
            d_real = discriminator(originImage, preImg)
        d_real_loss = costD(d_real, real_label)

        if model == 'SRGAN':
            d_fake = discriminator(preImg)
        elif model == 'ESRGAN':
            d_fake = discriminator(preImg, originImage)
        d_fake_loss = costD(d_fake, fake_label)
        d_total = d_real_loss + d_fake_loss
        d_total.backward()
        optD.step()
        DlossList.append(d_total.item())

        #训练Ｇ网络
        optG.zero_grad()
        g_real = generaotr(xnImage)
        if model == 'SRGAN':
            g_fake = discriminator(g_real)
        elif model == 'ESRGAN':
            g_fake = discriminator(g_real, originImage)
        gan_loss = costD(g_fake, real_label)
        mse_loss = costG(g_real, originImage)
        g_total = mse_loss + 1e-3 * gan_loss
        g_total.backward()
        optG.step()
        GlossList.append(g_total.item())

    plt.figure()
    plt.plot(GlossList)
    plt.title('Generator loss figure')
    plt.savefig(
        os.path.join(
            root, 'figure/{}_epoch_{}.png'.format(model + '_Generator',
                                                  epoch)))
    plt.figure()
    plt.plot(DlossList)
    plt.title('discriminator loss figure')
    plt.savefig(
        os.path.join(
            root, 'figure/{}_epoch_{}.png'.format(model + '_discriminator',
                                                  epoch)))
    GLoss = sum(GlossList) / len(GlossList)
    DLoss = sum(DlossList) / len(DlossList)
    print('Training loss of epoch {} Gloss is {} Dloss is {}'.format(
        epoch, GLoss, DLoss))

    generaotr.eval()
    discriminator.eval()
    validLoader_t = tqdm(validLoader)
    validLoader_t.set_description_str(' Valid epoch {}'.format(epoch))
    lossList = []
    for originImage, xnImage in validLoader_t:
        originImage = originImage.to(device)
        xnImage = xnImage.to(device)
        with torch.no_grad():
            preImage = generaotr(xnImage)
        loss = costG(preImage, originImage)
        lossList.append(loss)

    ValidLoss = sum(lossList) / len(lossList)
    print('Validing loss of epoch {} is {}'.format(epoch, ValidLoss))

    if ValidLoss < MinValidLoss or GLoss < MinGLoss or DLoss < MinDLoss:
        MinDLoss = min(MinDLoss, DLoss)
        MinGLoss = min(MinGLoss, GLoss)
        MinValidLoss = min(MinValidLoss, ValidLoss)
        print('Saving {} model......'.format(model))
        torch.save(generaotr.state_dict(),
                   os.path.join(root, 'models/{}_{}.pkl'.format(model, n)))
        torch.save(
            discriminator.state_dict(),
            os.path.join(
                root, 'models/{}_{}.pkl'.format(model + '_discriminator', n)))
    else:
        print('Valid loss is too large to save model......')
