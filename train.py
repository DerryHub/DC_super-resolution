import torch
from torch import nn
from torch import optim
from dataset import MyDataset
from net.carn.carn import CARN
from net.carn.carn_m import CARN_M
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

root = os.path.dirname(__file__)

useCUDA = True
model = 'CARN_M'
lr = 1e-3
EPOCH = 100
batch_size = 3
loadModel = True
n = 2

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

if model == 'CARN':
    net = CARN(n=n)
elif model == 'CARN_M':
    net = CARN_M(n=n)

print('Using {}'.format(model))

if useCUDA:
    net = net.cuda()

if not os.path.exists(os.path.join(root, 'models/')):
    os.mkdir(os.path.join(root, 'models/'))

if loadModel and os.path.exists(
        os.path.join(root, 'models/{}_{}.pkl'.format(model, n))):
    print('Loading {} model......'.format(model))
    net.load_state_dict(
        torch.load(os.path.join(root, 'models/{}_{}.pkl'.format(model, n))))

cost = nn.MSELoss(reduction='sum')

opt = optim.Adam(net.parameters(), lr=lr)

for epoch in range(EPOCH):
    trainLoader_t = tqdm(trainLoader)
    trainLoader_t.set_description_str(' Train epoch {}'.format(epoch))
    lossList = []
    net.train()
    for originImage, xnImage in trainLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            xnImage = xnImage.cuda()
        preImage = net(xnImage)
        loss = cost(preImage, originImage)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lossList.append(loss)

    plt.figure()
    plt.plot(lossList)
    plt.title('loss figure')
    plt.savefig(
        os.path.join(root, 'figure/{}_epoch_{}.png'.format(model, epoch)))

    print('Training loss of epoch {} is {}'.format(
        epoch,
        sum(lossList) / len(lossList)))

    net.eval()
    validLoader_t = tqdm(validLoader)
    validLoader_t.set_description_str(' Valid epoch {}'.format(epoch))
    lossList = []
    for originImage, xnImage in validLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            xnImage = xnImage.cuda()
        with torch.no_grad():
            preImage = net(xnImage)
        loss = cost(preImage, originImage)
        lossList.append(loss)

    print('Validing loss of epoch {} is {}'.format(
        epoch,
        sum(lossList) / len(lossList)))

    print('Saving {} model......'.format(model))
    torch.save(net.state_dict(),
               os.path.join(root, 'models/{}_{}.pkl'.format(model, n)))
