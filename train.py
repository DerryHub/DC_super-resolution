import torch
from torch import nn
from torch import optim
from dataset import MyDataset
from net.carn.carn import CARN
from net.carn.carn_m import CARN_M
from net.edsr.edsr import EDSR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

root = os.path.dirname(__file__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
useCUDA = True
model = 'CARN'
lr = 1e-4
EPOCH = 20
batch_size = 1
loadModel = False
n = 4

trainLoader = DataLoader(
    MyDataset(train=True, n=n),
    batch_size=batch_size,
    shuffle=True,
    num_workers=6)

# validLoader = DataLoader(
#     MyDataset(train=False, n=n),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=6)

if model == 'CARN':
    net = CARN(n=n)
elif model == 'CARN_M':
    net = CARN_M(n=n)
elif model == 'EDSR':
    net = EDSR(n=n)

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

# cost = nn.MSELoss(reduction='sum')
cost = nn.L1Loss(reduction='sum')

opt = optim.Adam(net.parameters(), lr=lr)

MinValidLoss = np.inf
MinTrainLoss = np.inf

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

    trainLoss = sum(lossList) / len(lossList)
    print('Training loss of epoch {} is {}'.format(epoch, trainLoss))

    net.eval()
    validDataset = MyDataset(train=False, n=n)
    validLoader_t = tqdm(validDataset)
    validLoader_t.set_description_str(' Valid epoch {}'.format(epoch))
    lossList = []
    for originImage, xnImage in validLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            xnImage = xnImage.cuda()
        originImage = torch.unsqueeze(originImage, 0)
        xnImage = torch.unsqueeze(xnImage, 0)
        with torch.no_grad():
            preImage = net(xnImage)
        loss = cost(preImage, originImage)
        lossList.append(loss)

    validLoss = sum(lossList) / len(lossList)
    print('Validing loss of epoch {} is {}'.format(epoch, validLoss))

    if validLoss < MinValidLoss or trainLoss < MinTrainLoss:
        MinTrainLoss = min(MinTrainLoss, trainLoss)
        MinValidLoss = min(MinValidLoss, validLoss)
        print('Saving {} model......'.format(model))
        torch.save(net.state_dict(),
                   os.path.join(root, 'models/{}_{}.pkl'.format(model, n)))
    else:
        print('Loss is too large to save model......')
    print()
