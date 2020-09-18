import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.transform import resize


sep = os.sep

im = Image.open(r'H:\深度学习理论与实战PyTorch实现\05.卷积神经网络（进阶）\资料\05.basic_conv download\basic_conv\cat.png').convert('L') # 读入一张灰度图的图片
im = np.array(im, dtype='float32')  # 将其转换为一个矩阵
im = resize(im, (60, 60), 3)
plt.imshow(im.astype('uint8'), cmap='gray')
plt.show()


# 将图片矩阵转化为 pytorch tensor，并适配卷积输入的要求
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
im = im.cuda()

# 使用 nn.Conv2d
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv3d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv3d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv3d(in_channel, 64, 5, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool3d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool3d(3)
        )

        self.classifier = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        xx = x
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        if self.verbose:
            print('flatten: {}'.format(x.shape))
        x1 = self.classifier(x)
        x2 = self.classifier2(x)
        return [x1, x2, xx]

test_net = resnet(3, 10, True)

test_x = Variable(torch.zeros(3, 3, 80, 80, 80))
test_y, test_y2, feature = test_net(test_x)

print('output: {}'.format(test_y.shape))
print('output: {}'.format(test_y2.shape))
print('output: {}'.format(feature.shape))












