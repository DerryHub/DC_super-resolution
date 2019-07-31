import torch
from torch import nn
from torch import optim
from dataset import MyDataset
import net.esrgan.esrgan as ESR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.autograd import Variable

root = os.path.dirname(__file__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
useCUDA = True
model = 'ESRGAN'
lr = 2e-4
EPOCH = 20
batch_size = 1
loadModel = False
n = 4

trainLoader = DataLoader(
    MyDataset(train=True, n=n),
    batch_size=batch_size,
    shuffle=True,
    num_workers=6)

if model == 'ESRGAN':
    generator = ESR.Generator(n=n, num=4)
    discriminator = ESR.Discriminator()
    # featureExtractor = ESR.FeatureExtractor()
    # featureExtractor.load_state_dict(
    #     torch.load(os.path.join(root, 'models/FeatureExtractor.pkl')))
    # featureExtractor.eval()

if useCUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    # featureExtractor = featureExtractor.cuda()

if not os.path.exists(os.path.join(root, 'models/')):
    os.mkdir(os.path.join(root, 'models/'))

if loadModel:
    print('Loading {} model......'.format(model))
    generator.load_state_dict(
        torch.load(os.path.join(root, 'models/{}_{}.pkl'.format(model, n))))
    discriminator.load_state_dict(
        torch.load(
            os.path.join(root, 'models/{}_{}_discriminator.pkl'.format(
                model, n))))

cost_GAN = nn.BCEWithLogitsLoss()
cost_content = nn.L1Loss()
cost_pixel = nn.L1Loss()

opt_G = optim.Adam(generator.parameters(), lr=lr)
opt_D = optim.Adam(discriminator.parameters(), lr=lr)

if not loadModel:
    print('pretraining generator......')
    trainLoader_t = tqdm(trainLoader)
    trainLoader_t.set_description_str(' pretrain...')
    for originImage, xnImage in trainLoader_t:
        if useCUDA:
            originImage = originImage.cuda()
            xnImage = xnImage.cuda()
        gen_hr = generator(xnImage)
        loss = cost_pixel(originImage, gen_hr)
        opt_G.zero_grad()
        loss.backward()
        opt_G.step()

print('Training......')

MinValidLoss = np.inf
MinTrainLoss_G = np.inf
MinTrainLoss_D = np.inf

torch.cuda.empty_cache()

for epoch in range(EPOCH):
    trainLoader_t = tqdm(trainLoader)
    trainLoader_t.set_description_str(' Train epoch {}'.format(epoch))
    lossList_G = []
    lossList_D = []
    generator.train()
    for originImage, xnImage in trainLoader_t:
        valid = Variable(
            torch.Tensor(np.ones((xnImage.size(0), 1, 64, 64))),
            requires_grad=False)
        fake = Variable(
            torch.Tensor(np.zeros((xnImage.size(0), 1, 64, 64))),
            requires_grad=False)
        if useCUDA:
            originImage = originImage.cuda()
            xnImage = xnImage.cuda()
            valid = valid.cuda()
            fake = fake.cuda()

        gen_hr = generator(xnImage)
        loss_pixel = cost_pixel(originImage, gen_hr)

        pred_real = discriminator(originImage).detach()
        pred_fake = discriminator(gen_hr)
        loss_GAN = cost_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # gen_features = featureExtractor(gen_hr)
        # real_features = featureExtractor(originImage).detach()
        # loss_content = cost_content(gen_features, real_features)

        loss_G = 5e-3 * loss_GAN + 1e-2 * loss_pixel

        lossList_G.append(loss_G)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        pred_real = discriminator(originImage)
        pred_fake = discriminator(gen_hr.detach())

        loss_real = cost_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = cost_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        lossList_D.append(loss_D)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

    plt.figure()
    plt.plot(lossList_D)
    plt.title('loss D figure')
    plt.savefig(
        os.path.join(root, 'figure/{}_epoch_{}_D.png'.format(model, epoch)))
    plt.figure()
    plt.plot(lossList_G)
    plt.title('loss G figure')
    plt.savefig(
        os.path.join(root, 'figure/{}_epoch_{}_G.png'.format(model, epoch)))

    trainLoss_G = sum(lossList_G) / len(lossList_G)
    trainLoss_D = sum(lossList_D) / len(lossList_D)
    print('Training loss G of epoch {} is {}'.format(epoch, trainLoss_G))
    print('Training loss D of epoch {} is {}'.format(epoch, trainLoss_D))

    generator.eval()
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
            preImage = generator(xnImage)
        loss = cost_pixel(preImage, originImage)
        lossList.append(loss)

    validLoss = sum(lossList) / len(lossList)
    print('Validing loss of epoch {} is {}'.format(epoch, validLoss))

    if validLoss < MinValidLoss or trainLoss_D < MinTrainLoss_D or trainLoss_G < MinTrainLoss_G:
        MinTrainLoss_D = min(MinTrainLoss_D, trainLoss_D)
        MinTrainLoss_G = min(MinTrainLoss_G, trainLoss_G)
        MinValidLoss = min(MinValidLoss, validLoss)
        print('Saving {} model......'.format(model))
        torch.save(generator.state_dict(),
                   os.path.join(root, 'models/{}_{}.pkl'.format(model, n)))
        torch.save(
            discriminator.state_dict(),
            os.path.join(root, 'models/{}_{}_discriminator.pkl'.format(
                model, n)))
    else:
        print('Loss is too large to save model......')
    print()