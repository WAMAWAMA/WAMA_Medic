import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from dice_loss import SoftDiceLoss,GDiceLossV2
# from torchsummary import summary
import time
import math

sep = os.sep


def char_color(s,front,word):
    """
    # 改变字符串颜色的函数
    :param s: 
    :param front: 
    :param word: 
    :return: 
    """
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char



class Index(object):
    def __init__(self, number=50, decimal=2):
        """
        :param decimal: 你保留的保留小数位
        :param number: # 号的 个数
        """
        self.decimal = decimal
        self.number = number
        self.a = 100 / number  # 在百分比 为几时增加一个 # 号

    def __call__(self, now, total):
        # 1. 获取当前的百分比数
        percentage = self.percentage_number(now, total)

        # 2. 根据 现在百分比计算
        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)

        # 3. 打印字符进度条
        progress_bar_num = self.progress_bar(well_num)

        # 4. 完成的进度条
        result = "\r%s %s" % (progress_bar_num, percentage)
        return result

    def percentage_number(self, now, total):
        """
        计算百分比
        :param now:  现在的数
        :param total:  总数
        :return: 百分
        """
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        """
        显示进度条位置
        :param num:  拼接的  “#” 号的
        :return: 返回的结果当前的进度条
        """
        # 1. "#" 号个数
        well_num = ">" * num


        # 2. 空格的个数
        space_num = " " * (self.number - num)

        return char_color('[%s%s]' % (well_num, space_num),50,32)


def conv3x3x3(in_channel, out_channel, stride=1):
    return nn.Conv3d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channel)

        self.conv2 = conv3x3x3(out_channel, out_channel)
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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up2(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=3, stride=stride, padding=1)

        self.conv = DoubleConv(in_channels//2+out_channels, (in_channels//2+out_channels))

    def forward(self, x1, x2):
        # print('@bfu x1 shape', x1.size())
        x1 = self.up(x1)
        # print('@afu x1 shape', x1.size())
        # print('@    x2 shape', x2.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # print('afp x1 shape', x1.size())
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2], dim=1)
        # print('@cancate shape', x.size())
        return self.conv(x)

        return x1

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=3, stride=stride,padding=1)

        self.conv = DoubleConv(in_channels//2+out_channels, (in_channels//2+out_channels)//2)

    def forward(self, x1, x2):
        # print('@bfu x1 shape', x1.size())
        x1 = self.up(x1)
        # print('@afu x1 shape', x1.size())
        # print('@    x2 shape', x2.size())

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # print('afp x1 shape', x1.size())
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2], dim=1)
        # print('@cancate shape', x.size())
        return self.conv(x)

        return x1

class resUnet(nn.Module):
    def __init__(self, in_channel=1, verbose=False):
        super(resUnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv3d(in_channel, 64, 3, 2, padding=1)

        self.block2 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        # def 2
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128),
            residual_block(128, 128),
            # residual_block(128, 128),
            # residual_block(128, 128),
            # residual_block(128, 128),
            # residual_block(128, 128),
            residual_block(128, 128)
        )

        # def 3
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256),
            residual_block(256, 256),
            residual_block(256, 256),
            residual_block(256, 256),
            # residual_block(256, 256),
            # residual_block(256, 256),
            # residual_block(256, 256),
            # residual_block(256, 256),
            # residual_block(256, 256),
            residual_block(256, 256)
        )

        # default 4
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            residual_block(512, 512),
            residual_block(512, 512),
            residual_block(512, 512),
            # residual_block(512, 512),
            # residual_block(512, 512),
            # residual_block(512, 512),
            # residual_block(512, 512),
            # residual_block(512, 512),
            # residual_block(512, 512),
            residual_block(512, 512),
            residual_block(512, 512),
            residual_block(512, 512)
        )

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128, 2)
        self.up3 = Up(128, 64, 2)
        self.up4 = Up2(64, 64, 2)

        self.outblock1 = nn.Sequential(
            nn.ConvTranspose3d(96, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            # ,  # 将尺寸还原到输出
            # nn.Conv3d(1, 1, 3, 1, padding=1)   # 再过一层卷积，可能有精修的作用
        )

        self.avg = nn.AvgPool3d(3)
        self.classifier1 = nn.Linear(512, 2)  # 是否是结节
        self.classifier2 = nn.Linear(512, 2)  # 是否大于3mm
        self.classifier3 = nn.Linear(512, 3)  # textrue
        self.classifier4 = nn.Linear(512, 4)  # F-score

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_()  # 全连接层参数初始化




    def forward(self, x):
        if self.verbose:
            print('input: {}'.format(x.shape))
        x1 = self.block1(x)
        if self.verbose:
            print('x1 output: {}'.format(x1.shape))
        x2 = self.block2(x1)
        if self.verbose:
            print('x2 output: {}'.format(x2.shape))
        x3 = self.block3(x2)
        if self.verbose:
            print('x3 output: {}'.format(x3.shape))
        x4 = self.block4(x3)
        if self.verbose:
            print('x4 output: {}'.format(x4.shape))
        x5 = self.block5(x4)
        if self.verbose:
            print('x5 output: {}'.format(x5.shape))


        UP = self.up1(x5, x4)
        if self.verbose:
            print('u4 output: {}'.format(UP.shape))

        UP = self.up2(UP, x3)
        if self.verbose:
            print('u3 output: {}'.format(UP.shape))

        UP = self.up3(UP, x2)
        if self.verbose:
            print('u2 output: {}'.format(UP.shape))

        UP = self.up4(UP, x1)
        if self.verbose:
            print('u1 output: {}'.format(UP.shape))

        UP = self.outblock1(UP)
        if self.verbose:
            print('u1 output: {}'.format(UP.shape))

        UP = torch.sigmoid(UP)

        # GAP
        avgg = self.avg(x5)

        # squeeze
        flatten = avgg.view(avgg.shape[0], -1)
        if self.verbose:
            print('flatten: {}'.format(flatten.shape))

        cls1 = self.classifier1(flatten)
        cls2 = self.classifier2(flatten)
        cls3 = self.classifier3(flatten)
        cls4 = self.classifier4(flatten)
        return [cls1, cls2, cls3, cls4, UP]





if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    test_net = resUnet(1, True).to(device)

    bz = 29
    test_x = Variable(torch.zeros(bz, 1, 80, 80, 80)).to(device)
    pred1, pred2, pred3, mask_pre = test_net(test_x)
    mask_pre.shape
    pred1.shape

    target1 = torch.randint(3, (bz,), dtype=torch.int64).to(device)
    target2 = torch.randint(3, (bz,), dtype=torch.int64).to(device)
    target3 = torch.randint(4, (bz,), dtype=torch.int64).to(device)

    loss1_smooth = SmoothCEloss(pred1, target1, C=3, alpha=0, mean=True, printflag=True)
    loss2_smooth = SmoothCEloss(pred2, target2, C=3, alpha=0, mean=True, printflag=True)
    loss3_smooth = SmoothCEloss(pred3, target3, C=4, alpha=0, mean=True, printflag=True)


    mask_pre = torch.squeeze(mask_pre, dim=1)
    diceloss = tverskydiceLoss(mask_pre,mask_pre,meanflg = True)


    loss_smooth = loss1_smooth+loss2_smooth+loss3_smooth+diceloss


    optimizer.zero_grad()
    loss_smooth.backward()
    optimizer.step()







    # 使用这个loss不需要squeeze，直接输入sigmoid后的即可，（b，1，x,y,z）
    citeraton = LovaszSoftmax()
    citeraton(mask_pre,0*mask_pre)






    # up = test_net(test_x)
    losser = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(test_net.parameters(), lr=0.01)


    loss1 = losser(pred1, target1)
    loss1_smooth = SmoothCEloss(pred1, target1, C=3, alpha=0, mean=True, printflag=True)
    print(loss1, loss1_smooth)

    # get_acc的使用
    output = F.softmax(pred1, dim=1)
    total = output.shape[0]  # batchsize
    _, pred_label = output.max(1)
    num_correct = int((pred_label == target1).sum().cpu().numpy())

    get_acc(pred1, target1)


    target2 = torch.randint(3, (bz,), dtype=torch.int64).to(device)
    loss2 = losser(pred2, target2)
    loss2_smooth = SmoothCEloss(pred2, target2, C=3, alpha=0, mean=True, printflag=True)
    print(loss2, loss2_smooth)


    target3 = torch.randint(4, (bz,), dtype=torch.int64).to(device)
    loss3 = losser(pred3, target3)
    loss3_smooth = SmoothCEloss(pred3, target3, C=4, alpha=0, mean=True, printflag=True)
    print(loss3, loss3_smooth)


    loss = loss1+loss2+loss3
    loss_smooth = loss1_smooth+loss2_smooth+loss3_smooth+diceloss
    loss.backward()
    loss_smooth.backward()


    print()
    print('output: {}'.format(pred1.shape))
    print('output: {}'.format(pred2.shape))
    print('output: {}'.format(pred3.shape))
    print('output: {}'.format(mask.shape))

    # from unet_model import UNet
    #
    # test_net = UNet(3, 10, True)
    #
    # test_x = Variable(torch.zeros(1, 3, 80, 80))
    # test_net(test_x)

    optimizer.zero_grad()
    loss_smooth.backward()
    optimizer.step()




    pred_mask = Variable(torch.tensor([0.5, 1., 0.5, 1.,0.5, 1.,0.5, 1.]))
    pred_mask.size()
    pred_mask = torch.reshape(pred_mask, [1, 2, 2, 2])
    pred_mask = torch.cat([pred_mask, pred_mask],dim=0)
    pred_mask.size()



    mask = Variable(torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]))
    mask.size()
    mask = torch.reshape(mask, [1, 2, 2, 2])
    mask = torch.cat([mask, mask], dim=0)
    mask.size()


    dice = tverskyDice(pred_mask,mask)
    diceloss = tverskydiceLoss(mask_pre,mask_pre,meanflg = True)
    ndice = get_dice(pred_mask,mask)



    optimizer.zero_grad()
    diceloss.backward()
    optimizer.step()


