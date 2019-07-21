import torch
from torch import nn
from torch import optim
from dataset import MyDataset
from net.carn.carn import CARN
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

root = os.path.dirname(__file__)

useCUDA = True
model = 'CARN'
lr = 1e-3
EPOCH = 10
batch_size = 8
loadModel = True

trainLoader = DataLoader(
    MyDataset(train=True), batch_size=batch_size, shuffle=True, num_workers=6)
validLoader = DataLoader(
    MyDataset(train=False),
    batch_size=batch_size,
    shuffle=False,
    num_workers=1)

if model == 'CARN':
    net = CARN()

print('Using {}'.format(model))

if useCUDA:
    net = net.cuda()

if not os.path.exists(os.path.join(root, 'models/')):
    os.mkdir(os.path.join(root, 'models/'))

if loadModel and os.path.exists(
        os.path.join(root, 'models/{}.pkl'.format(model))):
    print('Loading {} model......'.format(model))
    net.load_state_dict(
        torch.load(os.path.join(root, 'models/{}.pkl'.format(model))))

cost = nn.MSELoss(reduction='sum')

opt = optim.Adam(net.parameters(), lr=lr)

for epoch in range(EPOCH):
    trainLoader_t = tqdm(trainLoader)
    trainLoader_t.set_description_str(' Train epoch {}'.format(epoch))
    lossList = []
    net.train()
    for originImage, x4Image in trainLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            x4Image = x4Image.cuda()
        preImage = net(x4Image)
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
    for originImage, x4Image in validLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            x4Image = x4Image.cuda()
        with torch.no_grad():
            preImage = net(x4Image)
        loss = cost(preImage, originImage)
        lossList.append(loss)

    print('Validing loss of epoch {} is {}'.format(
        epoch,
        sum(lossList) / len(lossList)))

    print('Saving {} model......'.format(model))
    torch.save(net.state_dict(),
               os.path.join(root, 'models/{}.pkl'.format(model)))
