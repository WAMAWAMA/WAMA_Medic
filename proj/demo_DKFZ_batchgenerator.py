# 来自3D的尝试,首先读取数据
from wama.utils import *
img_path = r'D:\git\testnini\s22_v1.nii.gz'
mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'
subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square')

# show3D((bbox_image))
# show3Dslice(bbox_image)
bbox_image_batch = np.expand_dims(np.stack([bbox_image,bbox_image,bbox_image]),axis=1)
bbox_mask_batch = np.zeros(bbox_image_batch.shape)
bbox_mask_batch[:,:,20:100,20:100,20:100] = 1
bbox_image_batch = {'data':bbox_image_batch,'seg':bbox_mask_batch,'some_other_key':'as'}


# 构建transformer
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2,SpatialTransform
tr_transforms = []
# tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
deformation_scale = 0.2  # 0几乎没形变，0.2形变很大，故0~0.25是合理的
# （这个SpatialTransform_2与SpatialTransform的区别就在这里，SpatialTransform_2提供了有一定限制的扭曲变化，保证图像不会过度扭曲）
tr_transforms.append(
    SpatialTransform_2(
        patch_size=bbox_image.shape,
        patch_center_dist_from_border=[i // 2 for i in bbox_image.shape],
        do_elastic_deform=True, deformation_scale=(deformation_scale, deformation_scale+0.1),
        do_rotation=False,
        angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
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



bbox_image_batch_trans = all_transforms(**bbox_image_batch)
# bbox_image_batch_trans1 = all_transforms(**bbox_image_batch)

# show3D(np.concatenate([np.squeeze(bbox_image_batch['data'][0],axis=0),np.squeeze(bbox_image_batch_trans['data'][1],axis=0)],axis=1))
# show3Dslice(np.concatenate([np.squeeze(bbox_image_batch['data'][0],axis=0),np.squeeze(bbox_image_batch_trans['data'][1],axis=0)],axis=1))
# show3Dslice(np.concatenate([np.squeeze(bbox_image_batch['seg'][0],axis=0),np.squeeze(bbox_image_batch_trans['seg'][1],axis=0)],axis=1))
show3D(np.concatenate([np.squeeze(bbox_image_batch['seg'][0],axis=0),np.squeeze(bbox_image_batch_trans['seg'][1],axis=0)],axis=1))
# show3D(np.concatenate([np.squeeze(bbox_image_batch_trans1['data'][0],axis=0),np.squeeze(bbox_image_batch_trans['data'][0],axis=0)],axis=1))
# show3Dslice(np.concatenate([np.squeeze(bbox_image_batch_trans1['data'][0],axis=0),np.squeeze(bbox_image_batch_trans['data'][0],axis=0)],axis=1))







import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data

from batchgenerators.dataloading.data_loader import DataLoaderBase


class DataLoader(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE=2, num_batches=None, seed=False):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_batches, seed)
        # data is now stored in self._data.

    def generate_train_batch(self):
        # usually you would now select random instances of your data. We only have one therefore we skip this
        img = self._data

        # The camera image has only one channel. Our batch layout must be (b, c, x, y). Let's fix that
        img = np.tile(img[None, None], (self.BATCH_SIZE, 1, 1, 1))

        # now construct the dictionary and return it. np.float32 cast because most networks take float
        return {'data': img.astype(np.float32), 'some_other_key': 'some other value'}

batchgen = DataLoader(data.camera(), 4, None, False)
batch = next(batchgen)

def plot_batch(batch):
    batch_size = batch['data'].shape[0]
    plt.figure(figsize=(16, 10))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(batch['data'][i, 0], cmap="gray") # only grayscale image here
    plt.show()

plot_batch(batch)





from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import RndTransform
# if __name__ == '__main__':
#     from multiprocessing import freeze_support
#
#     freeze_support()
my_transforms = []
from batchgenerators.transforms.spatial_transforms import SpatialTransform

spatial_transform = SpatialTransform(data.camera().shape, np.array(data.camera().shape) // 2,
                 do_elastic_deform=True, alpha=(13000., 15000.), sigma=(30., 50.),
                 do_rotation=True, angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.3, 3.),
                 border_mode_data='constant', border_cval_data=0, order_data=1,
                 random_crop=False)

my_transforms.append(spatial_transform)
all_transforms = Compose(my_transforms)


# multithreaded_generator = MultiThreadedAugmenter(batchgen, all_transforms, 1, 2, seeds=None)
# multithreaded_generator = SingleThreadedAugmenter(batchgen, all_transforms)

batch_tmp = next(batchgen)

# plot_batch(multithreaded_generator.next1())
plot_batch(batch_tmp)

batch_tmp_trans = all_transforms(**batch_tmp)
plot_batch(batch_tmp_trans)
