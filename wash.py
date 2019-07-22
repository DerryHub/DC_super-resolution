from PIL import Image
import os
from tqdm import tqdm

root = os.path.dirname(__file__)

if not os.path.exists(os.path.join(root, 'DC_data/trainData')):
    os.mkdir(os.path.join(root, 'DC_data/trainData'))

i = 801

savePath = 'DC_data/train_raw/'

trainPath_1 = 'DC_data/RawData/DIV2K-dataset/DIV2K_train_HR/'
trainPath_2 = 'DC_data/RawData/DIV2K-dataset/DIV2K_valid_HR/'
trainPath_3 = 'DC_data/RawData/Flickr2K/Flickr2K_HR/'

imgName_1 = os.listdir(os.path.join(root, trainPath_1))
imgName_2 = os.listdir(os.path.join(root, trainPath_2))
imgName_3 = os.listdir(os.path.join(root, trainPath_3))

# for name in tqdm(imgName_1):
#     img = Image.open(os.path.join(root, trainPath_1, name))
#     a, b = img.size
#     m1 = min([a, b])
#     m2 = max([a, b])
#     while m1 > 1000:
#         m1 = m1 // 2
#         m2 = m2 // 2
#     m1 = m1 - m1%4
#     m2 = m2 - m2%4
#     if a > b:
#         img = img.resize((m2, m1), Image.BICUBIC)
#     else:
#         img = img.resize((m1, m2), Image.BICUBIC)
#     img.save(os.path.join(root, savePath, '{:0>4}.png'.format(i)))
#     i += 1

# for name in tqdm(imgName_2):
#     img = Image.open(os.path.join(root, trainPath_2, name))
#     a, b = img.size
#     m1 = min([a, b])
#     m2 = max([a, b])
#     while m1 > 1000:
#         m1 = m1 // 2
#         m2 = m2 // 2
#     m1 = m1 - m1 % 4
#     m2 = m2 - m2 % 4
#     print(name)
#     if a > b:
#         img = img.resize((m2, m1), Image.BICUBIC)
#     else:
#         img = img.resize((m1, m2), Image.BICUBIC)
#     img.save(os.path.join(root, savePath, '{:0>4}.png'.format(i)))
#     i += 1

for name in tqdm(imgName_3):
    img = Image.open(os.path.join(root, trainPath_3, name))
    a, b = img.size
    m1 = min([a, b])
    m2 = max([a, b])
    while m1 > 1000:
        m1 = m1 // 2
        m2 = m2 // 2
    m1 = m1 - m1 % 4
    m2 = m2 - m2 % 4
    if a > b:
        img = img.resize((m2, m1), Image.BICUBIC)
    else:
        img = img.resize((m1, m2), Image.BICUBIC)
    img.save(os.path.join(root, savePath, '{:0>4}.png'.format(i)))
    i += 1