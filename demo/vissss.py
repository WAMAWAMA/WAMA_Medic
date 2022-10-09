from wama.utils import *
from wama.data_augmentation import aug3D

img_path = r'D:\git\testnini\s22_v1.nii.gz'
mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'

subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square', aim_shape=[128,128,128])


bbox_image_batch = np.expand_dims(np.stack([bbox_image]*16),axis=1)# 构建batch
bbox_mask_batch = np.zeros(bbox_image_batch.shape)
bbox_mask_batch[:,:,20:100,20:100,20:100] = 1

bbox_image_batch = bbox_image_batch[:,:,:32,:,:]
bbox_mask_batch = bbox_mask_batch[:,:,:32,:,:]
auger = aug3D(size=[32,128,128], deformation_scale = 0.25) # size为原图大小即可（或batch大小）
aug_result = auger.aug(dict(data=bbox_image_batch, seg = bbox_mask_batch))  # 注意要以字典形式传入

# 可视化
index = 1
show3D(np.concatenate([np.squeeze(aug_result['seg'][index],axis=0),np.squeeze(bbox_mask_batch[index],axis=0)],axis=1))
aug_img = np.squeeze(aug_result['data'][index],axis=0)
show3D(np.concatenate([aug_img,bbox_image],axis=1)*100)

pth = r"C:\1225827_Han Li Hua_tumor2.pkl"
case = load_from_pkl(pth)
show3Dslice(np.concatenate([mat2gray(adjustWindow(case['img'], 321, 123)), case['mask']],1))
show3D(np.concatenate([mat2gray(adjustWindow(case['img'], 321, 123)), case['mask']],1))
show3D(case['mask'])

# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# bagging = BaggingClassifier(KNeighborsClassifier(),...                             max_samples=0.5, max_features=0.5)








