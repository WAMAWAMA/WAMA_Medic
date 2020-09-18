import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from model import resUnet
import random
from datetime import datetime
# from torchsummary import summary
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from scipy.ndimage import zoom
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
import h5py
import SimpleITK as sitk
from model import Index as indexbar
from lovasz_loss import LovaszSoftmax
from dice_loss import DC_and_CE_loss,GDiceLossV2,GDiceLoss,FocalTversky_loss
from torch.optim import lr_scheduler
from mdUnet import Modified3DUNet
from skimage import transform
from skimage import util
from scipy.ndimage import zoom
import ctypes
from preprocessing.augmentation import augment3DImage
import cv2
import preprocessing.utils as preutils
from step1_raw2nii import readCsv

import pandas as pd
import seaborn as sn
from tensorboardX import SummaryWriter




sep = os.sep


def save_cfmx(pre_label, true_label, savepath):
    data = {'y_Predicted': pre_label, 'y_Actual': true_label}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)

    confusion_matrix_img = sn.heatmap(confusion_matrix, annot=True, fmt='.20g')
    fig = confusion_matrix_img.get_figure()
    fig.savefig(savepath)  # 路径+文件名
    plt.close()


def showDataMask(scan_cube,mask_cube):
    # Display mid slices from resampled scan/mask
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:], cmap=plt.cm.gray)
    axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
    axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:], cmap=plt.cm.gray)
    axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
    axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)], cmap=plt.cm.gray)
    axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])
    plt.show()


def load_h5(filepath,indexlist):
    data = []
    H5_file = h5py.File(filepath, 'r')
    for index in indexlist:
        data.append(H5_file[index][:])
    H5_file.close()
    return data


def aug_scale(data,mask):
    """
    只放大
    :param data: 
    :param mask: 
    :return: 
    """
    scalesize = random.uniform(0.9, 1.2)
    sh = data.shape
    cube_size = int(sh[0]*scalesize)
    data = zoom(data, (cube_size / sh[0], cube_size / sh[1], cube_size / sh[2]), order=2)  # resample for cube_size
    mask = zoom(mask, (cube_size / sh[0], cube_size / sh[1], cube_size / sh[2]), order=1)  # resample for cube_size
    return [data,mask]


def aug_rotate(data, mask, dim = 0, rotation = None):
    """
    随机沿着dim轴旋转,角度随机在0~180度内
    :param data: 
    :param mask: 
    :return: 
    """
    # data fomat(x,y,z)
    # 随机三个平面三个角度旋转
    if rotation is None:
        rotation = int(random.uniform(0., 1.)*180)


    # 先把mask切换为float,然后在转回去
    mask = np.array(mask,dtype=np.float)



    # rotate by skimage =============================================
    # - 0: Nearest - neighbor
    # - 1: Bi - linear(default)
    # - 2: Bi - quadratic
    # - 3: Bi - cubic
    # - 4: Bi - quartic
    # - 5: Bi - quintic
    # data_inter_method = 3
    # mask_inter_method = 3
    # if dim == 0:
    #     for i in range(data.shape[dim]):
    #         data[i,:,:] = transform.rotate(data[i,:,:], rotation, order=data_inter_method)
    #         mask[i,:,:] = transform.rotate(mask[i,:,:], rotation, order=mask_inter_method)
    # elif dim == 1:
    #     for i in range(data.shape[dim]):
    #         data[:,i,:] = transform.rotate(data[:,i,:], rotation, order=data_inter_method)
    #         mask[:,i,:] = transform.rotate(mask[:,i,:], rotation, order=mask_inter_method)
    # elif dim == 2:
    #     for i in range(data.shape[dim]):
    #         data[:,:,i] = transform.rotate(data[:,:,i], rotation, order=data_inter_method)
    #         mask[:,:,i] = transform.rotate(mask[:,:,i], rotation, order=mask_inter_method)


    # rotate by cv2(will be faster than skimage)


    if dim == 0:
        for i in range(data.shape[dim]):
            data[i,:,:] = preutils.rotate_image(data[i,:,:], rotation, cv2.INTER_LINEAR)
            mask[i,:,:] = preutils.rotate_image(mask[i,:,:], rotation, cv2.INTER_LINEAR)
    elif dim == 1:
        for i in range(data.shape[dim]):
            data[:,i,:] = preutils.rotate_image(data[:,i,:], rotation, cv2.INTER_LINEAR)
            mask[:,i,:] = preutils.rotate_image(mask[:,i,:], rotation, cv2.INTER_LINEAR)
    elif dim == 2:
        for i in range(data.shape[dim]):
            data[:,:,i] = preutils.rotate_image(data[:,:,i], rotation, cv2.INTER_LINEAR)
            mask[:,:,i] = preutils.rotate_image(mask[:,:,i], rotation, cv2.INTER_LINEAR)




    # mask[mask > 0.2] = 1
    # mask = np.array(mask,dtype=np.uint8)


    return data,mask


def aug_noise(data):
    if random.uniform(0., 1.)<0.8:
        for i in range(data.shape[0]):
            data[i,:,:] = util.random_noise(data[i,:,:], mode='gaussian',var=0.001)
        return data
    else:
        return data


def aug_changedim(data, mask, dim=None):
    """
    随机调换轴的顺序,或指定顺序
    :param data: 
    :param mask: 
    :return: 
    """
    if dim is None:
        dim = [0,1,2]
        random.shuffle(dim)
    data = np.transpose(data,dim)
    mask = np.transpose(mask,dim)
    return [data,mask]


def aug_mixup(data):
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        # Mixup images.
        lambda_ = beta_distribution.sample([]).item()
        index = torch.randperm(images.size(0)).cuda()
        mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

        # Mixup loss.
        scores = model(mixed_images)
        loss = (lambda_ * loss_function(scores, labels)
                + (1 - lambda_) * loss_function(scores, labels[index]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return data


def LNDb_filename2label(filename):
    # exp:r'2_6_1_1_0_2_3.h5'
    filename = filename.split('.')[0]
    labels = filename.split('_')
    LNDbID,FindingID,agrlevel,isNodule,isOver3mm,Textlevel,F = [int(i) for i in labels]
    return [LNDbID,FindingID,agrlevel,isNodule,isOver3mm,Textlevel,F]


def get_filelist_frompath4LNDb(filepath, expname, sample_id=None):
    """
    
    :param filepath: 
    :param expname: 
    :param sample_id: 是个id的list,LNDbID 即可
    :return: 
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            # print(file)
            if file.endswith('.' + expname):
                # 获得样本id
                [id, _, _, _, _, _, _] = LNDb_filename2label(file)
                # 如果id在sample_id里,那么就吧这个path加入
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def CT_adjust_window(data,CTWW,CTWL):
    """
    
    :param data: CT 数据
    :param HU_low:  
    :param HU_high: 
    :return: 
    """
    data = np.array(data,dtype=np.float)
    HU_high = CTWL+0.5*CTWW
    HU_low = CTWL-0.5*CTWW


    data[data<HU_low]=HU_low
    data[data>HU_high]=HU_high
    data_scale = (data-HU_low)/(HU_high-HU_low)

    return data_scale


def dataloader4train(filelist, batchsize=16, mixup=False, aug=True, aug_p = 0.75, cube_size = 80):
    """

    :param file_path:
    :param batchsize:
    :param mixup: 是否使用mixup
    :param aug: 是否使用扩增,主要是旋转
    :param aug_p: 随机扩增的概率
    :return:
    """
    # 读取列表
    iter = 0
    epoch = 1
    index = 0

    # 设置窗宽窗位
    # CTWW,CTWL = [1700,-500]
    CTWW,CTWL = [1700,-600]



    # 数据格式:3D 图片(b,c,x,y,z) mask(同图片),label(b,classes_num),label非one-hot
    data_container = np.zeros([batchsize, 1, cube_size, cube_size, cube_size], dtype=np.float32)
    mask_container = np.zeros([batchsize, 1, cube_size, cube_size, cube_size], dtype=np.float32)
    label1 = np.zeros([batchsize, 1], dtype=np.float32)  # isNodule(结节1) 二分类问题,考虑减少神经元为1后使用BCEloss
    label2 = np.zeros([batchsize, 1], dtype=np.float32)  # isOver3mm(大于1) 二分类问题,同上
    label3 = np.zeros([batchsize, 1], dtype=np.float32)  # Textlevel(0,1,2,3),3代表非结节病变,包括则为4分类,不包括则为3分类
    label4 = np.zeros([batchsize, 1], dtype=np.float32)  # fisher分数,4分类

    while(True):
        iter += 1
        # 确定随机扩增的概率
        truep = random.uniform(0., 1.)
        if aug and truep < aug_p:
            print('aug')
        for i in range(batchsize):
            # 读取batch =============================================================================================
            filepath = filelist[index]
            # 读取CT 和 mask
            data,mask = load_h5(filepath,['data','mask'])
            # 获取label
            [_, _, _, isNodule, isOver3mm, Textlevel, F] = LNDb_filename2label(filepath.split(sep)[-1])


            # 随机窗宽窗位,以达到数据扩增的作用
            if aug and truep<aug_p:
                realCTWW = random.uniform(0.88, 1.12)*CTWW
                realCTWL = random.uniform(0.88, 1.12)*CTWL
            else:
                realCTWW = CTWW
                realCTWL = CTWL

            data = CT_adjust_window(data,realCTWW,realCTWL)
            # showDataMask(data,mask)

            if aug and truep<aug_p:  # 这里除了对床款创维扩增外(相当于对比度),主要就是调换dim顺序和多轴旋转
                # print('use aug')
                # 随机调换dim顺序
                data, mask = aug_changedim(data, mask)
                # 1个维度旋转(多次旋转会因为插值产生问题)
                data, mask = aug_rotate(data, mask, dim = 0)
                # 随机加高斯噪声
                data = aug_noise(data)
                # 随机方法(没有缩小,sorry)
                # data, mask = aug_scale(data, mask)
            # showDataMask(data, mask)
            # 扩增完或旋转完,裁剪到cube_size
            if np.max(data.shape)>cube_size:
                length = data.shape[0]
                res = (length-cube_size)//2
                data = data[res:res+cube_size,res:res+cube_size,res:res+cube_size]
                mask = mask[res:res+cube_size,res:res+cube_size,res:res+cube_size]



            data_container[i,0,:,:,:] = data
            mask_container[i,0,:,:,:] = mask
            label1[i,:] = isNodule
            label2[i,:] = isOver3mm
            label3[i,:] = Textlevel
            label4[i,:] = F
            index += 1
            if index == len(filelist):
                index = 0
                epoch = epoch + 1
                random.shuffle(filelist)  # new

            # ============================================================================

        # 这里可以变成tensor再返回去
        # if mixup:
        #     print('use mixup')

        yield [iter, epoch, data_container, mask_container, label1, label2, label3, label4]


def dataloader4test(filelist, batchsize, cube_size=80):
    """
    测试函数应该返回文件列表或者id,以供记录用
    :param filelist: 
    :param batchsize: 
    :param mixup: 
    :param aug: 
    :return: 
    """

    # 设置窗宽窗位
    CTWW,CTWL = [1700,-500]


    # 计算一些参数
    testtset_num = len(filelist)
    indexxx = 0
    all_iters = testtset_num // batchsize  # 一共要迭代predict的次数
    res_num = testtset_num % batchsize  # 不能纳入完整batch的剩余样本数


    for iii in range(all_iters):
        # 如果是最后一个batch,那么就吧剩下的也加上
        if iii == all_iters - 1:
            real_batch_size = batchsize + res_num
        else:
            real_batch_size = batchsize

        # 构建容器
        data_container = np.zeros([real_batch_size, 1, cube_size, cube_size, cube_size], dtype=np.float32)
        mask_container = np.zeros([real_batch_size, 1, cube_size, cube_size, cube_size], dtype=np.float32)
        label1 = np.zeros([real_batch_size, 1], dtype=np.float32)  # isNodule(结节1) 二分类问题,考虑减少神经元为1后使用BCEloss
        label2 = np.zeros([real_batch_size, 1], dtype=np.float32)  # isOver3mm(大于1) 二分类问题,同上
        label3 = np.zeros([real_batch_size, 1], dtype=np.float32)  # Textlevel(0,1,2,3),3代表非结节病变,包括则为4分类,不包括则为3分类
        label4 = np.zeros([real_batch_size, 1], dtype=np.float32)  # fisher分数,4分类

        LNDbID_list = []
        FindingID_list = []

        print(iii+1,all_iters)

        for i in range(real_batch_size):
            # 读取batch =============================================================================================
            filepath = filelist[indexxx]
            # 读取CT 和 mask
            data,mask = load_h5(filepath,['data','mask'])
            # 获取label
            [LNDbID, FindingID, _, isNodule, isOver3mm, Textlevel, F] = LNDb_filename2label(filepath.split(sep)[-1])
            LNDbID_list.append(LNDbID)
            FindingID_list.append(FindingID)


            # 预处理
            data = CT_adjust_window(data,CTWW,CTWL)

            # 裁剪到80
            if np.max(data.shape)>cube_size:
                length = data.shape[0]
                res = (length-cube_size)//2
                data = data[res:res+cube_size,res:res+cube_size,res:res+cube_size]
                mask = mask[res:res+cube_size,res:res+cube_size,res:res+cube_size]

            # 放入容器
            data_container[i,0,:,:,:] = data
            mask_container[i,0,:,:,:] = mask
            label1[i,:] = isNodule
            label2[i,:] = isOver3mm
            label3[i,:] = Textlevel
            label4[i,:] = F
            indexxx += 1
            # ============================================================================

        # 这里可以变成tensor再返回去,返回的第一位为是否结束读取位
        if indexxx != len(filelist):
            yield [True, LNDbID_list, FindingID_list, data_container, mask_container, label1, label2, label3, label4]
        else:
            yield [False, LNDbID_list, FindingID_list, data_container, mask_container, label1, label2, label3, label4]


def SmoothCEloss(logits, labels, C=2, alpha=0.2,mean = False, printflag = False):
    """

    :param logits:
    :param labels: 不用独热编码
    :param C: class num
    :param alpha: smooth parameter
    :return:
    """
    N = labels.size(0)  # batchsize
    if printflag:
        print('batchsize:', N)


    smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).to(logits.device)   # 注意这个to,hhhhh
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)
    if printflag:
        print('@or     label', '\n', labels)
        print('@smooth label', '\n', smoothed_labels)

    log_prob = torch.nn.functional.log_softmax(logits, dim=1)

    loss = -torch.sum(log_prob * smoothed_labels, dim=[0, 1]) / N  # mean_loss of the batch
    loss_no_mean = -torch.sum(log_prob * smoothed_labels, dim=1)  # per ins loss
    if mean:
        return loss
    else:
        return loss_no_mean


def tverskyDice(pred,target):
    # beta越大，敏感度越高
    # pred是经过sigmoid的概率图，不是logits
    # target是与pred同样大小的二值图
    # 格式（b, x, y, z）, 不要有channel维度，有的话需要在外面squeeze掉

    alpha = 0.5
    beta = 0.5
    p1 = 1-pred
    g1 = 1-target
    num = (pred * target).sum(dim=(1,2,3))
    # print('num',num,num.shape)
    den = num + alpha * (pred * g1).sum(dim=(1,2,3)) + beta * (p1 * target).sum(dim=(1,2,3))
    # print('den',den,den.shape)
    T = num/(den+1e-9)
    # print('T',T.shape)
    return T


def tverskydiceLoss(pred, target, nonSquared=False,meanflg = True):
    # pred是经过sigmoid的概率图，不是logits
    # target是与pred同样大小的二值图
    # 格式（b, x, y, z）, 不要有channel维度，有的话需要在外面squeeze掉
    if meanflg:
        return (1 - tverskyDice(pred, target)).mean(dim=0)
    else:
        return 1 - tverskyDice(pred, target)


def normalDice(pred, target, thresold=0.5,meanflg = True):
    """
    # pred是经过sigmoid的概率图，!!!不是logits
    # target是与pred同样大小的二值图
    # 格式（b, x, y, z）, 不要有channel维度，有的话需要在外面squeeze掉

    :param pred: 经过sigmoid的概率图
    :param target: 与pred同样大小的二值图
    :param thresold: 
    :param meanflg: 
    :return: 
    """


    # 卡个阈值变成二值图
    pred = torch.gt(pred, thresold).float()
    # print('pred', pred)

    # 求交集
    num = (pred * target).sum(dim=(1,2,3))
    # print(num.shape)
    # print('num', num)

    # 求并集
    den = (pred + target).sum(dim=(1,2,3))
    # print('den', den)

    # 计算并返回dice
    if meanflg:
        return ((2*num + 1e-9)/(den+1e-9)).mean(dim=0)
    else:
        return (2*num + 1e-9)/(den+1e-9)


def get_acc(preprob, label):
    """

    :param preprob: logits就行，（b，y）
    :param label:
    :return:
    """
    output = F.softmax(preprob, dim=1)
    total = output.shape[0]  # batchsize
    _, pred_label = output.max(1)

    print(label)
    print(pred_label)
    num_correct = int((pred_label == label).sum().cpu().numpy())
    return num_correct / total


def get_dice(output, label, thresold=0.5, mean = False):
    """

    :param output: 预测的sigmoid概率图,格式（b,x,y,z）,不要有channel维度，有的话需要在外面squeeze掉
    :param label:
    :param thresold:
    :param mean:
    :return:
    """
    return normalDice(output, label, thresold=thresold,meanflg = mean)


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


def view_dataset4LNDb(filelist, view = False):
    """
    查看数据集各个样本情况的函数,返回多标签中各个类别的比例
    :param filelist: 
    :return: 
    """
    # LNDbID =    [LNDb_filename2label(i.split(sep)[-1])[0] for i in filelist]
    # FindingID = [LNDb_filename2label(i.split(sep)[-1])[1] for i in filelist]
    agrlevel =  [LNDb_filename2label(i.split(sep)[-1])[2] for i in filelist]
    isNodule =  [LNDb_filename2label(i.split(sep)[-1])[3] for i in filelist]
    isOver3mm = [LNDb_filename2label(i.split(sep)[-1])[4] for i in filelist]
    Textlevel = [LNDb_filename2label(i.split(sep)[-1])[5] for i in filelist]
    Fscore =    [LNDb_filename2label(i.split(sep)[-1])[6] for i in filelist]


    # LNDbID_class = np.unique(LNDbID)
    # FindingID_class = np.unique(FindingID)
    agrlevel_class = np.unique(agrlevel)
    isNodule_class = np.unique(isNodule)
    isOver3mm_class = np.unique(isOver3mm)
    Textlevel_class = np.unique(Textlevel)
    F_class = np.unique(Fscore)

    agrlevel_class_num = [agrlevel.count(i) for i in agrlevel_class]
    isNodule_class_num = [isNodule.count(i) for i in isNodule_class]
    isOver3mm_class_num = [isOver3mm.count(i) for i in isOver3mm_class]
    Textlevel_class_num = [Textlevel.count(i) for i in Textlevel_class]
    F_class_num = [Fscore.count(i) for i in F_class]

    if view:
        print('===============================================')
        print('count',len(agrlevel))
        print('class index ',2,char_color('agrlevel_class',50,32),agrlevel_class,char_color('perclass_num',50,31),agrlevel_class_num)
        print('class index ',3,char_color('isNodule_class',50,32),isNodule_class,char_color('perclass_num',50,31),isNodule_class_num)
        print('class index ',4,char_color('isOver3mm_class',50,32),isOver3mm_class,char_color('perclass_num',50,31),isOver3mm_class_num)
        print('class index ',5,char_color('Textlevel_class',50,32),Textlevel_class,char_color('perclass_num',50,31),Textlevel_class_num)
        print('class index ',6,char_color('F_class',50,32),F_class,char_color('perclass_num',50,31),F_class_num)
        print('===============================================')

    return [None,
            None,
            [list(agrlevel_class), agrlevel_class_num],
            [list(isNodule_class), isNodule_class_num],
            [list(isOver3mm_class), isOver3mm_class_num],
            [list(Textlevel_class), Textlevel_class_num],
            [list(F_class), F_class_num]]


def class_balance4LDNb(filelist,class_index):
    """
    单类别平衡函数
    :param file_list: 
    :param class_index: 制定的类别标签,3~6,分别对应如下
    3 - isNodule
    4 - isOver3mm
    5 - Textlevel
    6 - Fscore
    :return: 某个类别平衡后的list
    """

    ins_class_list = [LNDb_filename2label(i.split(sep)[-1])[class_index] for i in filelist]
    classes, classnum = view_dataset4LNDb(filelist)[class_index]
    max_class_index = classnum.index(np.max(classnum))
    max_class_ins_num = classnum[max_class_index]

    new_filelist = []
    for needbalanceclass in classes:
        if needbalanceclass is not max_class_index:
            # 首先把对应的needbalanceclass的filelist拿出来
            needbalanceclass_filelist = []
            for index, ins in enumerate(ins_class_list):
                if ins == needbalanceclass:
                    needbalanceclass_filelist.append(filelist[index])

            # 求和最大类别的差距
            res_NUM = max_class_ins_num - len(needbalanceclass_filelist)
            for i in range(res_NUM):
                new_filelist.append(random.sample(needbalanceclass_filelist, 1)[0])


    return new_filelist+filelist


def getacc4array(pre,label):
    tmp1 = (pre - label)
    tmp1[tmp1 != 0] = 1
    acc = 1 - ((np.sum(tmp1) / len(tmp1)))
    return acc


def getfoldLNDb(foldcsv,foldlist=[1,2,3]):
    """
    
    :param foldcsv: 
    :param foldlist: 读取第一折,则为[1],读取2,3,4折,则为[2,3,4]
    :return: 
    """
    CTcsvlines = readCsv(foldcsv)
    header = CTcsvlines[0]
    nodules = CTcsvlines[1:]

    LNDb_list = []

    for fold in foldlist:
        for n in nodules:
            LNDb_list.append(int(n[header.index('Fold'+str(fold))]))
    return LNDb_list

# from:https://github.com/ildoonet/pytorch-gradual-warmup-lr
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)



if __name__ == "__main__":

    writer = SummaryWriter()
    prev_time = datetime.now()

    # 参数设置
    foldcsv = r'/media/root/老王/@data_LNDb/LNDb dataset/trainset_csv/trainFolds.csv'
    # model_savepath = r'/data/@data_laowang/data4resUnet/model'
    result_savepath = r'/data/@data_laowang/data4resUnet/result_bigLR'
    # train_path = r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114/over3'
    # test_path = r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114/over3'
    max_epoch = 500
    init_lr = 1e-2
    min_lr = 1e-3
    # train_batchsize = 20
    train_batchsize = 20
    test_batchsize = 10
    iter = 0
    GPU = 2  # 指定GPU
    loss_weight = [1.,1.,1.,1.,1.]  # isNodule, isOver3mm, Textlevel, fisher分数, seg
    augflag = True

    warm_up_epoch = 20
    cos_decay_epoch = 200
    # lr_decay_ratio = 0.8

    # 设备准备
    device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")

    # 文件夹准备
    result_savepath_index = result_savepath+sep+'index'
    result_savepath_image = result_savepath+sep+'image'
    os.system('mkdir ' + result_savepath_index)
    os.system('mkdir ' + result_savepath_image)


    # 分折信息读取
    train_fold_id = getfoldLNDb(foldcsv,foldlist=[1,2,3])
    test_fold_id = getfoldLNDb(foldcsv,foldlist=[0])




    # 读取列表

    # train_list = get_filelist_frompath4LNDb(train_path,'h5')
    # train_list1 = get_filelist_frompath4LNDb(r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114_newnew(bigger)/less3','h5')
    # train_list2 = get_filelist_frompath4LNDb(r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114_newnew(bigger)/non','h5')
    train_list = get_filelist_frompath4LNDb(r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114_newnew(bigger)/over3','h5',train_fold_id)


    test_list = get_filelist_frompath4LNDb(r'/data/@data_laowang/data4resUnet/@data_nodule(80multisqrt2)114_newnew(bigger)/over3','h5',test_fold_id)

    random.shuffle(train_list)
    print(len(train_list))



    # 平衡样本
    print('trainset detail:')
    view_dataset4LNDb(train_list, view = True)
    train_list = class_balance4LDNb(train_list,5)
    view_dataset4LNDb(train_list, view=True)

    random.shuffle(train_list)
    random.shuffle(train_list)


    print('testset detail:')
    view_dataset4LNDb(test_list, view=True)



    # 构建loader
    train_loader = dataloader4train(train_list, train_batchsize,cube_size=80,aug = augflag)
    test_loader = dataloader4test(test_list, test_batchsize,cube_size=80)

    # 构建模型
    net = resUnet().to(device)
    # net = Modified3DUNet(in_channels=1,n_classes=1).to(device)

    # 设置优化器
    # optimizer = torch.optim.SGD(net.parameters(), init_lr)  # 使用随机梯度下降，学习率 0.1
    optimizer = torch.optim.Adam(net.parameters(), min_lr)  # 使用随机梯度下降，学习率 0.1



    # 学习率策略
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.88)
    scheduler_cos = lr_scheduler.CosineAnnealingLR(optimizer, cos_decay_epoch, eta_min=min_lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=init_lr/min_lr, total_epoch=warm_up_epoch, after_scheduler=scheduler_cos)


    # 实例化loss类(如果有的话)
    citeraton1 = LovaszSoftmax()
    citeraton2 = GDiceLoss()
    citeraton3 = GDiceLossV2()
    citeraton4 = FocalTversky_loss()  # 这个貌似确实会好很多


    # 开始训练
    per_epoch_contain_batch = (len(train_list)//train_batchsize)+1
    bar = indexbar(number=50)
    lr_list = []
    label_mask_saved = False
    for epoch in range(max_epoch):
        net = net.train()
        # 每个epoch的train阶段 =================================================
        for iter_batch in range(per_epoch_contain_batch):
            # a = 1

            tic = datetime.now()

            print(bar(iter_batch + 1, per_epoch_contain_batch),'%', end=' ')
            print(' iter:',char_color(str(iter),50,31),
                  ' epoch:',char_color(str(epoch+1),50,31),'/',max_epoch,
                  ' batch:',char_color(str(iter_batch+1),50,31),'/',per_epoch_contain_batch)
            iter += 1


            # forward =========================================================
            # 加载数据
            [_, _, data, mask, label1, label2, label3, label4] = next(train_loader)
            # 转移到设备
            data = torch.from_numpy(data).float().to(device)
            mask = torch.from_numpy(mask).float().to(device)
            label1 = torch.from_numpy(label1).long().to(device)
            label2 = torch.from_numpy(label2).long().to(device)
            label3 = torch.from_numpy(label3).long().to(device)
            label4 = torch.from_numpy(label4).long().to(device)

            label1 = torch.squeeze(label1, dim=1)
            label2 = torch.squeeze(label2, dim=1)
            label3 = torch.squeeze(label3, dim=1)
            label4 = torch.squeeze(label4, dim=1)

            # feed
            pred1, pred2, pred3, pred4, pre_mask_prob = net(data)

            # 计算loss
            loss_seg1 = citeraton4(pre_mask_prob, mask)


            pre_mask_prob = torch.squeeze(pre_mask_prob, dim=1)
            mask = torch.squeeze(mask, dim=1)

            # loss_cls1 = SmoothCEloss(pred1, label1, C=2, alpha=0, mean=True, printflag=False)
            # loss_cls2 = SmoothCEloss(pred2, label2, C=2, alpha=0, mean=True, printflag=False)
            loss_cls3 = SmoothCEloss(pred3, label3, C=3, alpha=0.05, mean=True, printflag=False)  # 不算非结节,纹理只分3类
            # loss_cls3 = SmoothCEloss(pred4, label4, C=4, alpha=0, mean=True, printflag=False)
            # loss_seg1 = tverskydiceLoss(pre_mask_prob,mask,meanflg = True)

            # final_loss = loss_weight[0]*loss_cls1+\
            #              loss_weight[1]*loss_cls2+\
            #              loss_weight[2]*loss_cls3+\
            #              loss_weight[3]*loss_cls4+\
            #              loss_weight[4]*loss_seg1

            # final_loss = loss_cls1+ loss_cls2+loss_cls3+loss_cls4+loss_seg1
            # final_loss = loss_cls3+10*loss_seg1  # dice可以收敛到很高,训练集
            final_loss = 5*loss_cls3+10*loss_seg1
            # final_loss = 10*loss_seg1
            # final_loss = loss_cls3

            # backward =========================================================
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()


            # 记录下loss,acc,以及各种指标dice之类的,训练的时候一律取均值===========================
            # acc1 = get_acc(pred1, label1)
            # acc2 = get_acc(pred2, label2)
            acc3 = get_acc(pred3, label3)
            # acc4 = get_acc(pred4, label4)
            dice = get_dice(pre_mask_prob,mask,mean=True)

            print('acc:',acc3)
            print('dice: {:.3f}'.format(dice.data))

            print('seg_loss1: {:.3f}'.format(loss_seg1.data))
            print('cls3_loss: {:.3f}'.format(loss_cls3.data))

            dice = dice.cpu().detach().numpy()
            loss_seg1 = loss_seg1.cpu().detach().numpy()
            loss_cls3 = loss_cls3.cpu().detach().numpy()
            current_lr = optimizer.param_groups[0]['lr']
            print('LR',current_lr)
            # 记录训练集的指标到txt
            test_txt = open(result_savepath_index + sep + 'train_result.txt', 'a')
            test_txt.write(' @iter: ' + str(iter) +
                           ' @acc3: ' + str('%03f' % acc3) +
                           ' @dice: ' + str('%03f' % dice) +
                           ' @loss_cls3: ' + str('%03f' % loss_cls3) +
                           ' @loss_seg1: ' + str('%03f' % loss_seg1) +
                           ' @LR: ' + str(current_lr) +'\n')
            test_txt.close()

            # ================ 使用 tensorboard ===============
            writer.add_scalars('train acc3(iter)', {'train': acc3}, iter)
            writer.add_scalars('train dice(iter)', {'train': dice}, iter)
            writer.add_scalars('train loss_cls3(iter)', {'train': loss_cls3}, iter)
            writer.add_scalars('train loss_seg1(iter)', {'train': loss_seg1}, iter)
            writer.add_scalars('train LR(iter)', {'train': current_lr}, iter)
            # =================================================




            # 显示当前iter运行的时间
            toc = datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(time_str)


            #======================================================================

        lr_list.append(optimizer.param_groups[0]['lr'])
        # 学习率策略
        if epoch <= warm_up_epoch+cos_decay_epoch:
            scheduler.step()

        plt.plot(lr_list)


        # # 每个epoch的test阶段 ==================================================
        # # 可以挪倒train的每个epoch里面,实现一个epoch验证or测试多次
        not_end_flag = True
        net = net.eval()  # 设置状态
        test_pred1_list = []
        test_pred2_list = []
        test_pred3_list = []
        test_pred4_list = []

        test_pre_mask_prob_list = []
        test_mask_list = []

        test_dice_list = []

        test_LNDbID = []
        test_FindingID = []

        test_label1 = []
        test_label2 = []
        test_label3 = []
        test_label4 = []

        # 开始测试
        with torch.no_grad():
            while (not_end_flag):
                # 加载数据
                [not_end_flag, LNDbID_list, FindingID_list, data, mask, label1, label2, label3, label4]= next(test_loader)
                print(not_end_flag)
                test_LNDbID = test_LNDbID + LNDbID_list
                test_FindingID = test_FindingID + FindingID_list

                # 转移到设备
                data = torch.from_numpy(data).float().to(device)
                mask = torch.from_numpy(mask).float().to(device)
                label1 = torch.from_numpy(label1).long().to(device)
                label2 = torch.from_numpy(label2).long().to(device)
                label3 = torch.from_numpy(label3).long().to(device)
                label4 = torch.from_numpy(label4).long().to(device)

                label1 = torch.squeeze(label1, dim=1)
                label2 = torch.squeeze(label2, dim=1)
                label3 = torch.squeeze(label3, dim=1)
                label4 = torch.squeeze(label4, dim=1)

                test_label1.append(label1.cpu().detach().numpy())
                test_label2.append(label2.cpu().detach().numpy())
                test_label3.append(label3.cpu().detach().numpy())
                test_label4.append(label4.cpu().detach().numpy())


                # feed&predict
                pred1, pred2, pred3, pred4, pre_mask_prob = net(data)
                pred1 = F.softmax(pred1, dim=1)
                pred2 = F.softmax(pred2, dim=1)
                pred3 = F.softmax(pred3, dim=1)
                pred4 = F.softmax(pred4, dim=1)

                # 计算得到每个样本的dice
                pre_mask_prob = torch.squeeze(pre_mask_prob, dim=1)
                mask = torch.squeeze(mask, dim=1)
                testdice = get_dice(pre_mask_prob,mask,mean=False)


                # 储存指标到对应list:每个样本的dice以及对应预测值
                test_pred1_list.append(pred1.cpu().detach().numpy())
                test_pred2_list.append(pred2.cpu().detach().numpy())
                test_pred3_list.append(pred3.cpu().detach().numpy())
                test_pred4_list.append(pred4.cpu().detach().numpy())

                test_pre_mask_prob_list.append(pre_mask_prob.cpu().detach().numpy())
                test_mask_list.append(mask.cpu().detach().numpy())

                test_dice_list.append(testdice.cpu().detach().numpy())


        # 测试结束,计算指标并打印 ---------------------------------
        # 汇总
        final_pred1 = np.concatenate(test_pred1_list,axis=0)
        final_pred2 = np.concatenate(test_pred2_list,axis=0)
        final_pred3 = np.concatenate(test_pred3_list,axis=0)
        final_pred4 = np.concatenate(test_pred4_list,axis=0)
        final_dice = np.concatenate(test_dice_list,axis=0)

        test_label1 = np.concatenate(test_label1,axis=0)
        test_label2 = np.concatenate(test_label2,axis=0)
        test_label3 = np.concatenate(test_label3,axis=0)
        test_label4 = np.concatenate(test_label4,axis=0)

        test_pre_mask_prob = np.concatenate(test_pre_mask_prob_list, axis=0)
        test_mask_label = np.concatenate(test_mask_list, axis=0)



        # 计算acc等,dice等指标
        final_pred1_label = np.argmax(final_pred1, axis = 1)
        final_pred2_label = np.argmax(final_pred2, axis = 1)
        final_pred3_label = np.argmax(final_pred3, axis = 1)
        final_pred4_label = np.argmax(final_pred4, axis = 1)


        test_acc1 = getacc4array(final_pred1_label,test_label1)
        test_acc2 = getacc4array(final_pred2_label,test_label2)
        test_acc3 = getacc4array(final_pred3_label,test_label3)
        test_acc4 = getacc4array(final_pred4_label,test_label4)
        test_dice = np.mean(final_dice)

        print('test ===============================')
        print('epoch',epoch)
        # print('acc1',test_acc1)
        # print('acc2',test_acc2)
        print('acc3',test_acc3)
        # print('acc4',test_acc4)
        print('dice',test_dice)


        # 记录指标到txt
        test_txt = open(result_savepath_index+sep+'test_result.txt', 'a')
        test_txt.write(' @epoch: '+str(epoch)+
                       ' @acc1: '+str('%03f' % test_acc1)+
                       ' @acc2: '+str('%03f' % test_acc2)+
                       ' @acc3: '+str('%03f' % test_acc3)+
                       ' @acc4: '+str('%03f' % test_acc4)+
                       ' @dice: '+str('%03f' % test_dice)+'\n')
        test_txt.close()

        # 保存预测mask到文件中

        mask_save_folder = result_savepath_image + sep + str(epoch)
        os.system('mkdir ' + mask_save_folder)
        if label_mask_saved is not True:
            for iii in range(test_mask_label.shape[0]):
                tmp_mask =test_mask_label[iii,:,:]
                itkimage = sitk.GetImageFromArray(tmp_mask, isVector=False)
                tmp_filename = mask_save_folder+sep+str(test_LNDbID[iii])+'_'+str(test_FindingID[iii])+'gt.nii.gz'
                sitk.WriteImage(itkimage, tmp_filename, True)
            # 职位label_mask_saved
            label_mask_saved = True

        for iii in range(test_pre_mask_prob.shape[0]):
            tmp_mask = test_pre_mask_prob[iii,:,:]
            tmp_mask[tmp_mask>0.5] = 1
            tmp_mask[tmp_mask<=0.5] = 0
            itkimage = sitk.GetImageFromArray(tmp_mask, isVector=False)
            tmp_filename = mask_save_folder+sep+str(test_LNDbID[iii])+'_'+str(test_FindingID[iii])+'pre.nii.gz'
            sitk.WriteImage(itkimage, tmp_filename, True)

        # 保存混淆矩阵到文件夹中
        save_cfmx(final_pred3_label, test_label3, result_savepath_index+sep+str(epoch)+'_cfmx.png')

        # 结束test记得重新初始化testloader
        test_loader = dataloader4test(test_list, test_batchsize)
        # #======================================================================

        # ================ 使用 tensorboard ===============
        writer.add_scalars('test acc', {'test': test_acc3}, epoch)
        writer.add_scalars('test dice', {'test': test_dice}, epoch)
        # =================================================

    # 完整实验结束
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    print(time_str)





















