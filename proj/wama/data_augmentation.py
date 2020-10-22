import numpy as np

# import BG related lib
from batchgenerators.transforms.abstract_transforms import Compose as BG_Compose


# import PIL related lib
from PIL import Image
from torchvision import transforms as T

# import AG related lib
import imgaug as ia
from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)



"""扩增需要注意的"""
# batch_dict，字典，key为data和seg





"""2D 的扩增"""
# 2D 的扩增 --------------------------------------------------------------

"""batchgenerator的(简称BG）"""
# 自动考虑是否有seg
# 扩增概率已经内部实现，设置参数即可
# BG_transforms_seq = []
# BG_transforms_seq.append()
# BG_transforms = BG_Compose(BG_transforms_seq)
# def BG_augmenter(batch_dict):
#     """
#     :param batch_dict: batch_dict['data']和['seg']都应该包括c和bz轴，即（bz，c，w，h）
#     :return:
#     """
#     return BG_transforms(**batch_dict)




"""imgaug的(简称AG）""" # pass
# @ 可能和少楠的有一些不同
# 需要手动考虑是否有seg，
# 注意维度，AG的channel轴在最后一维，即batch的shape是（bz，w，h，c），所以需要调整batch_dict里面数据的dim order才能处理
# 扩增概率已经内部实现，设置参数即可
AG_transforms_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50%图像进行水平翻转
    iaa.Flipud(0.5),  # 50%图像做垂直翻转
    sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
    iaa.OneOf([  # 用高斯模糊，均值模糊，中值模糊中的一种增强
        iaa.GaussianBlur((0, 3.0)),
        iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
        iaa.MedianBlur(k=(3, 11)),]),
    sometimes(iaa.Affine(  # 对一部分图像做仿射变换
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
        rotate=(-45, 45),  # 旋转±45度之间
        shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
        order=[0, 1],  # 使用最邻近差值或者双线性差值
        cval=(0, 255),
        mode=ia.ALL,  # 边缘填充
    )),
    iaa.OneOf([  # 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    ]),
    sometimes(  # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    ),
],random_order=True)
AG_transforms = iaa.Sequential(AG_transforms_seq)
def img_aug_augmenter(batch_dict):
    """
    :param batch_dict: batch_dict['data']和['seg']都应该包括c和bz轴，即（bz，c，w，h）
    :return:
    """
    tmp_patch_data = batch_dict['data']  # ndarray with shape （bz，c，w，h）
    if 'seg' in batch_dict.keys():
        tmp_patch_mask = batch_dict['seg']  # if not None, ndarray with shape （bz，c，w，h）
    else:
        tmp_patch_mask = None


    # 调整维度，将（bz，c，w，h）调整为AG可以处理的（bz，w，h，c）
    tmp_patch_data = np.transpose(tmp_patch_data, (0, 2, 3, 1))
    if tmp_patch_mask is not None:
        tmp_patch_mask = np.transpose(tmp_patch_mask, (0, 2, 3, 1))



    if tmp_patch_mask is not None:
        tmp_patch_data_aug, tmp_patch_mask_aug = AG_transforms(images=tmp_patch_data, segmentation_maps=tmp_patch_mask)
    else:
        tmp_patch_data_aug = AG_transforms(images=tmp_patch_data)

    # 调整维度回去，将（bz，w，h，c）调整为（bz，c，w，h）
    tmp_patch_data_aug = np.transpose(tmp_patch_data_aug, (0, 3, 1, 2))
    if tmp_patch_mask is not None:
        tmp_patch_mask_aug = np.transpose(tmp_patch_mask_aug, (0, 3, 1, 2))


    # 返回
    if tmp_patch_mask is not None:
        return {'data': tmp_patch_data_aug, 'seg': tmp_patch_mask_aug}
    else:
        return {'data': tmp_patch_data_aug}









"""3D 的扩增"""
# 3D 的扩增 --------------------------------------------------------------
# 暂时只有DKFZ的BG




# import numpy as np
# import imgaug.augmenters as iaa
#
# # Standard scenario: You have N=16 RGB-images and additionally one segmentation
# # map per image. You want to augment each image and its heatmaps identically.
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
# segmaps = np.random.randint(0, 10, size=(16, 64, 64, 1), dtype=np.int32)
#
# seq = iaa.Sequential([
#     iaa.GaussianBlur((0, 3.0)),
#     iaa.Affine(translate_px={"x": (-40, 40)}),
#     iaa.Crop(px=(0, 10))
# ])
#
# images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)















