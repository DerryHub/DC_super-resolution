import torch
from torch import nn
from torch import optim
from dataset import MyDataset
from net.carn.carn import CARN
from torch.utils.data import DataLoader
from tqdm import tqdm

useCUDA = False
model = 'CARN'
lr = 1e-3
EPOCH = 1
batch_size = 1

trainLoader = DataLoader(MyDataset(train=True), batch_size=batch_size, shuffle=True, num_workers=1)
validLoader = DataLoader(MyDataset(train=False), batch_size=batch_size, shuffle=False, num_workers=1)

if model == 'CARN':
    net = CARN()

if useCUDA:
    net = net.cuda()

cost = nn.MSELoss()

opt = optim.Adam(net.parameters(), lr=lr)

for epoch in range(EPOCH):
    trainLoader_t = tqdm(trainLoader)
    for originImage, x4Image in trainLoader_t:
        trainLoader_t.set_description_str(' epoch {}'.format(epoch))
        if useCUDA:
            originImage = originImage.cuda()
            x4Image = x4Image.cuda()
        preImage = net(x4Image)
        loss = cost(preImage, originImage)
        opt.zero_grad()
        loss.backward()
        opt.step()
    