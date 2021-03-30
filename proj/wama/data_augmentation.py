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
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2,SpatialTransform


def get_transformer(bbox_image_shape = [256, 256, 256], deformation_scale = 0.2):
    """

    :param bbox_image_shape:  [256, 256, 256]
    :param deformation_scale: 扭曲程度，0几乎没形变，0.2形变很大，故0~0.25是合理的
    :return:
    """
    tr_transforms = []
    # tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    # （这个SpatialTransform_2与SpatialTransform的区别就在这里，SpatialTransform_2提供了有一定限制的扭曲变化，保证图像不会过度扭曲）

    tr_transforms.append(
        SpatialTransform_2(
            patch_size=bbox_image_shape,
            patch_center_dist_from_border=[i // 2 for i in bbox_image_shape],
            do_elastic_deform=True, deformation_scale=(deformation_scale, deformation_scale + 0.1),
            do_rotation=False,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),  # 随机旋转的角度
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=False,
            scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=False,
            p_el_per_sample=1.0, p_rot_per_sample=1.0, p_scale_per_sample=1.0
        ))
    # tr_transforms.append(
    #     SpatialTransform(
    #         patch_size=bbox_image.shape,
    #         patch_center_dist_from_border=[i // 2 for i in bbox_image.shape],
    #         do_elastic_deform=True, alpha=(2000., 2100.), sigma=(10., 11.),
    #         do_rotation=False,
    #         angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         do_scale=False,
    #         scale=(0.75, 0.75),
    #         border_mode_data='constant', border_cval_data=0,
    #         border_mode_seg='constant', border_cval_seg=0,
    #         order_seg=1, order_data=3,
    #         random_crop=True,
    #         p_el_per_sample=1.0, p_rot_per_sample=1.0, p_scale_per_sample=1.0
    #     ))
    # sigma越小，扭曲越局部（即扭曲的越严重）， alpha越大扭曲的越严重
    # tr_transforms.append(
    #     SpatialTransform(bbox_image.shape, [i // 2 for i in bbox_image.shape],
    #                      do_elastic_deform=True, alpha=(1300., 1500.), sigma=(10., 11.),
    #                      do_rotation=False, angle_z=(0, 2 * np.pi),
    #                      do_scale=False, scale=(0.3, 0.7),
    #                      border_mode_data='constant', border_cval_data=0, order_data=1,
    #                      border_mode_seg='constant', border_cval_seg=0,
    #                      random_crop=False))

    all_transforms = Compose(tr_transforms)
    return all_transforms

# demo
# bbox_image_batch_trans = all_transforms(**bbox_image_batch)  # 加入**相当于
# bbox_image_batch是一个字典，'data'和'seg'字段，形状为【bz，c，w，h，l】

class aug3D():
    def __init__(self, size = [256,256,256], deformation_scale = 0.2):
        self.trans = get_transformer(size, deformation_scale)
    def aug(self, batch_dict):
        """
        batch 中不同样本会经过不同的变换，即一个batch内样本间是不一样的变换
        batch_dict， data和seg字段，如下面的形状
        :param img: 【bz，c，w，h，l】
        :param seg: 【bz，c，w，h，l】
        :return:
        """
        img = batch_dict['data']
        if 'seg' in batch_dict.keys():
            seg = batch_dict['seg']
        else:
            seg = None


        if seg is not None:
            return self.trans(data = img, seg = seg)
        else:
            return self.trans(data = img)

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













if __name__ == '__main__':

    print('demo')

