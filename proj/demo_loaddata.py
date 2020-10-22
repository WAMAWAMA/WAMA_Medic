import imageio
import numpy as np
from wama.utils import mat2gray, show2D
from wama.data_augmentation import img_aug_augmenter
import datetime


# # 读取图片,制作dict
# img_pth = r'D:\git\detec_precess\proj\data_demo\segmentation_data\images\001.jpg'
# mask_pth = r'D:\git\detec_precess\proj\data_demo\segmentation_data\masks\001.PNG'
# img = np.array(imageio.imread(img_pth))
# mask = (np.array(imageio.imread(mask_pth)))
#
# img = np.stack([img,img,img,img,img])
# mask = np.stack([mask,mask,mask,mask,mask])
#
#
# input_dict = {'data':(np.transpose(img, [0,3,1,2])),
#               'seg': (np.expand_dims(mask, axis=1))}
# out_dict = img_aug_augmenter(input_dict)
# index = 1
# show2D(np.transpose((out_dict['data'][index,:,:,:]), [1, 2, 0]))
# show2D((np.squeeze(out_dict['seg'][index,:,:])))


# dataloader 使用例子
def readImgMask(img_pth, mask_pth):
    img = np.array(imageio.imread(img_pth))
    mask = np.array(imageio.imread(mask_pth))
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, [0, 3, 1, 2])
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return {'img': img, 'seg': mask}


def trans2normalimg(img, seg):
    img = np.squeeze(img)
    img = np.transpose(img, [1, 2, 0])
    seg = np.squeeze(seg, axis=0)
    seg = np.transpose(seg, [1, 2, 0])
    return {'img': img, 'seg': seg}


def labelonehot(all_class, class_index):
    """
    独热编码
    :param all_class: 所有类别数量
    :param class_index: 当前类别，从1开始
    :return:
    """
    tmp = np.zeros(all_class)
    tmp[class_index - 1] = 1
    return {'label1': tmp}


def labelonehot4label1(all_class, label1_class_index):
    return labelonehot(all_class, label1_class_index)


def dataaug(img, seg):
    datadict = {'data': img, 'seg': seg}
    output = img_aug_augmenter(datadict)
    return {'img': output['data'], 'seg': output['seg']}


# 通过搭建多条pipeline并行处理数据，可以使多任务读取数据更加灵活，每次增减pipeline即可
# 例子，暂时构建如下pipeline
# pipeline 1：读取数据和mask，扩增
# pipeline 2：读取分类 label 1，并实现one-hot encoding
# pipeline 4：传递 img_name, subject_id
pipeline1 = [
    {'method': readImgMask, 'in_key': ['img_pth', 'mask_pth'], 'out_key': ['img', 'seg']},
    {'method': dataaug, 'in_key': ['img', 'seg'], 'out_key': ['img', 'seg']},
    {'method': trans2normalimg, 'in_key': ['img', 'seg'], 'out_key': ['img', 'seg']},
    {'method': None, 'in_key': ['img', 'seg'], 'out_key': ['img', 'seg']},
    {'method': None, 'in_key': ['img', 'seg'], 'out_key': ['img', 'seg']},
]
pipeline2 = [
    {'method': labelonehot4label1,
     'in_key': ['all_class', 'label1_class_index'],
     'out_key': ['label1']},
]
pipeline3 = [
    {'method': None, 'in_key': ['img_pth', 'subject_id'], 'out_key': ['img_pth', 'subject_id']},
]
pipeline_list = [pipeline1, pipeline2, pipeline3]

from wama.data_loader_beta import chechout_pipeline, run_pipeline, wama_dataset, get_loader

indict = {
    'img_pth': r'D:\git\detec_precess\proj\data_demo\segmentation_data\images\001.jpg',
    'mask_pth': r'D:\git\detec_precess\proj\data_demo\segmentation_data\masks\001.PNG',
    'all_class': 5,
    'label1_class_index': 2,
    'subject_id': 1212
}
import matplotlib.pyplot as plt


def plot_batch(batch):
    batch_size = batch['seg'].shape[0]
    plt.figure(figsize=(16, 10))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(batch['img'][i])  # only grayscale image here
    plt.show()


a = run_pipeline(in_dict=indict, pipeline=pipeline1)
a = run_pipeline(in_dict=indict, pipeline=pipeline2)
a = run_pipeline(in_dict=indict, pipeline=pipeline3)

input_dict_list = [indict]
for _ in range(7):
    input_dict_list = input_dict_list + input_dict_list
print(len(input_dict_list))

dataloader = get_loader(input_dict_list=input_dict_list,
                        pipeline_list=pipeline_list,
                        num_workers=0,
                        pin_memory=False,
                        batch_size=4)

tic_ = datetime.datetime.now()
tic = datetime.datetime.now()
dataloader.__len__()
for i, sample in enumerate(dataloader):
    toc = datetime.datetime.now()
    print(i, '/', dataloader.__len__())
    # 计时结束
    h, remainder = divmod((toc - tic).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
    print((time_str))
    tic = toc
    # plot_batch(sample)

toc = datetime.datetime.now()
h, remainder = divmod((toc - tic_).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
print((time_str))

# plot_batch(sample)
