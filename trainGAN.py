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

root = os.path.dirname(__file__)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

costG = nn.MSELoss(reduction='sum')
costD = nn.BCELoss()

optG = optim.Adam(generaotr.parameters(), lr=lr, betas=(0.9, 0.999))
optD = optim.SGD(
    discriminator.parameters(), lr=lr / 100, momentum=0.9, nesterov=True)
steplrG = torch.optim.lr_scheduler.StepLR(optG, 5)
steplrD = torch.optim.lr_scheduler.StepLR(optD, 10)
validLoss = np.inf

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

        real_label = torch.ones(xnImage.size(0), xnImage.size(1)).to(device)
        fake_label = torch.zeros(xnImage.size(0), xnImage.size(1)).to(device)
        #训练Ｄ网络
        optD.zero_grad()
        d_real = discriminator(originImage)
        d_real_loss = costD(d_real, real_label)

        d_fake = discriminator(generaotr(xnImage))
        d_fake_loss = costD(d_fake, fake_label)
        d_total = d_real_loss + d_fake_loss
        d_total.backward()
        optD.step()
        DlossList.append(d_total.item())

        #训练Ｇ网络
        optG.zero_grad()
        g_real = generaotr(xnImage)
        g_fake = discriminator(g_real)
        gan_loss = costD(g_fake, real_label)
        mse_loss = costG(g_real, originImage)
        g_total = mse_loss + 1e-2 * gan_loss
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

    print('Training loss of epoch {} Gloss is {} Dloss is {}'.format(
        epoch,
        sum(GlossList) / len(GlossList),
        sum(DlossList) / len(DlossList)))

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

    print('Validing loss of epoch {} is {}'.format(
        epoch,
        sum(lossList) / len(lossList)))

    if sum(lossList) / len(lossList) < validLoss:
        validLoss = sum(lossList) / len(lossList)
        print('Saving {} model......'.format(model))
        torch.save(generaotr.state_dict(),
                   os.path.join(root, 'models/{}_{}.pkl'.format(model, n)))
        torch.save(
            discriminator.state_dict(),
            os.path.join(
                root, 'models/{}_{}.pkl'.format(model + '_discriminator', n)))
    else:
        print('Valid loss is too large to save model......')
