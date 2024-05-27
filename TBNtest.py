# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from PIL import Image
import torch
from torch.utils import data as Data
import torch.nn as nn
import math
import os

import warnings

warnings.filterwarnings('ignore')


class MyDataset(Dataset):
    def __init__(self, root, train=True):
        samples = []
        self.train = train

        samples.append((root, 'ckd'))
        self.samples = samples

        # transform manner is different between train and valid/test data
        if train:
            self.transform = T.Compose([
                # T.RandomRotation(degrees=1),
                T.Resize((224, 224)),
                T.RandomCrop(224, padding=2),
                T.ToTensor(),
                T.Normalize([0.34796497, 0.34796497, 0.34796497], [0.17704655, 0.17704655, 0.17704655])  # 3
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.34796497, 0.34796497, 0.34796497], [0.17704655, 0.17704655, 0.17704655])
            ])

    def __getitem__(self, index):
        '''
        return (img, texture_features, label) for each sample
        '''
        path, label = self.samples[index]
        gray = cv2.imread(path, 0)  # read as gray image
        texture_feature = get_glcm_norm(gray)
        # print(label)
        label = 1 if 'ckd' in label else 0
        img = Image.open(path)
        img = self.transform(img)

        return img, torch.Tensor(texture_feature), label, path

    def __len__(self):
        return len(self.samples)


def get_glcm_norm(image):
    image = cv2.resize(image, (224, 224))
    # image = exposure.rescale_intensity(image, in_range=(0, 255), out_range=(0, 63))
    glcm = greycomatrix(image, [1, 2, 3, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
    glcm[0, 0, :, :] = 0
    # glcm[0, :, :, :] = 0
    # glcm[:, 0, :, :] = 0
    glcm_feat = []  # 'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'
    # dataset1
    values = {
        'contrast': [910.91590909090928, 16.89604019859064, 278.32445754310965, 177.98475750300378],
        'dissimilarity': [21.323703445875374, 3.1835962524023058, 11.08635316368998, 3.7437287297031689],
        'homogeneity': [0.26872117478738927, 0.055619616870056308, 0.11026783477528417, 0.035246657647461904],
        'energy': [0.031429261249563412, 0.0087894603955558422, 0.015576096983089874, 0.0033147364182192626],
        'correlation': [0.99315117748065129, 0.56979840275574334, 0.90198791268758693, 0.067109718617275355]
    }
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
                 ]:
        fea = greycoprops(glcm, prop).flatten()
        mean = values[prop][2]
        std = values[prop][3]
        fea = (np.asarray(fea) - mean) / std
        glcm_feat.append(fea)

    return np.asarray(glcm_feat).flatten()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # texture branch
        self.tb_layer1 = nn.Sequential(
            nn.Linear(80, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.tb_layer2 = nn.Sequential(
            nn.Linear(80, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.tb_layer3 = nn.Sequential(
            nn.Linear(80, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        res = t
        out2 = self.tb_layer1(t)
        out2 = self.tb_layer2(out2)
        out2 = self.tb_layer3(out2)
        out2 = res + out2

        output = torch.cat([x, out2], 1)

        x = self.fc(output)

        return x


'''
def test():
    model.eval()
    with torch.no_grad():
        for cnt, (x, t, y, pth) in enumerate(test_loader, 1):
            if USE_GPU:
                x = x.cuda()
                t = t.cuda()
                y = y.cuda()
            # optimizer.zero_grad()
            output = model(x, t)
            _, predict = torch.max(output, 1)
            output = output.sigmoid().cpu().numpy()
            if predict.item() == 1:
                print('筛查结果为：慢性肾脏病')
            else:
                print('筛查结果为：正常')
            print('概率: {:.2f}%'.format(99*output[0,predict.item()])) 
'''


def judge(root):
    test_set = MyDataset(root=root, train=False)
    test_loader = Data.DataLoader(test_set, batch_size=1)
    print("Number of valid set:", len(test_set))

    # 创建基础模型
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    # 修改模型
    in_fea = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_fea + 80, 2)
    )
    # 读入模型参数
    model.load_state_dict(torch.load(os.path.join('static/parameters', 'glcm.pkl'), map_location=lambda storage, loc: storage))

    USE_GPU = False
    # USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        model = model.cuda()
        print('推理开启 GPU加速')
    else:
        print('推理使用 CPU')

    print('测试数据:{}'.format(root))

    # 测试部分
    result = True
    model.eval()
    with torch.no_grad():
        for cnt, (x, t, y, pth) in enumerate(test_loader, 1):
            if USE_GPU:
                x = x.cuda()
                t = t.cuda()
                y = y.cuda()
            output = model(x, t)
            _, predict = torch.max(output, 1)
            output = output.sigmoid().cpu().numpy()
            if predict.item() == 1:
                print('筛查结果为：慢性肾脏病')
            else:
                print('筛查结果为：正常')
                result = False
            print('概率: {:.2f}%'.format(99 * output[0, predict.item()]))
    return result


if __name__ == '__main__':
    root = './test1.jpg'
    judge(root)
