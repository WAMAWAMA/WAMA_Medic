import numpy as np

# import BG related lib
from batchgenerators.transforms.abstract_transforms import Compose as BG_Compose


# import PIL related lib
from PIL import Image
from torchvision import transforms as T

# import AG related lib
import imgaug as ia
from imgaug import augmenters as iaa




"""扩增需要注意的"""
# batch_dict，字典，key为data和seg





"""2D 的扩增"""
# 2D 的扩增 --------------------------------------------------------------

# batchgenerator的(简称BG）
# 自动考虑是否有seg
# 扩增概率已经内部实现，设置参数即可
BG_transforms_seq = []
BG_transforms_seq.append()
BG_transforms = BG_Compose(BG_transforms_seq)
def BG_augmenter(batch_dict):
    return BG_transforms(**batch_dict)



# PIL的（简称PIL）
# 需要手动考虑是否有seg, 且需要分别实现seg和img两个的变换（因为插值等问题）
# 注意维度，PIL的channel轴在最后一维，即batch的shape是（bz，w，h，c），所以需要调整batch_dict里面数据的dim order才能处#理
# 扩增概率需要手动实现
# 需要抓换为pil类型的，才能继续使用PIL处理
PIL_transforms_seq_img = []
PIL_transforms_seq_mask = []
PIL_transforms_seq_img.append()
PIL_transforms_seq_mask.append()
PIL_transforms_img = T.Compose(PIL_transforms_seq_img)
PIL_transforms_mask = T.Compose(PIL_transforms_seq_mask)
def PIL_augmenter(batch_dict):
    # 获得图片
    tmp_patch_data = batch_dict['data']  # ndarray with shape （bz，c，w，h）
    if 'seg' in batch_dict.keys():
        tmp_patch_mask = batch_dict['seg']  # if not None, ndarray with shape （bz，c，w，h）
    else:
        tmp_patch_mask = None


    # 调整维度，将（bz，c，w，h）调整为AG可以处理的（bz，w，h，c）
    tmp_patch_data = np.transpose(tmp_patch_data, (0, 2, 3, 1))
    if tmp_patch_mask is not None:
        tmp_patch_mask = np.transpose(tmp_patch_mask, (0, 2, 3, 1))

    # PIL不能对batch处理，故需要写个循环来处理
    for i in tmp_patch_data.shape[0]:
        # 取出单个sample
        _img = tmp_patch_data[i,:,:,:]
        if tmp_patch_mask is not None:
            _mask = tmp_patch_mask[i,:,:,:]

        #  转换为PIL独有的格式
        _img = Image.fromarray(_img)
        if tmp_patch_mask is not None:
            _mask = Image.fromarray(_mask)

        # 淦
        _img = PIL_transforms_img(_img)
        if tmp_patch_mask is not None:
            _mask = PIL_transforms_mask(_mask)

        # 转换回numpy，储存到原tmp_patch_data


    # 调整dim顺序




    # 返回

    raise NotImplementedError  # todo





# imgaug的(简称AG）
# 需要手动考虑是否有seg，
# 注意维度，AG的channel轴在最后一维，即batch的shape是（bz，w，h，c），所以需要调整batch_dict里面数据的dim order才能处理
# 扩增概率已经内部实现，设置参数即可
AG_transforms_seq = ([])
AG_transforms_seq.append()
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




import numpy as np
import imgaug.augmenters as iaa

# Standard scenario: You have N=16 RGB-images and additionally one segmentation
# map per image. You want to augment each image and its heatmaps identically.
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
segmaps = np.random.randint(0, 10, size=(16, 64, 64, 1), dtype=np.int32)

seq = iaa.Sequential([
    iaa.GaussianBlur((0, 3.0)),
    iaa.Affine(translate_px={"x": (-40, 40)}),
    iaa.Crop(px=(0, 10))
])

images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)















