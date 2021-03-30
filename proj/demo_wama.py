from wama.utils import *
from wama.utils import *


img_path = r'D:\git\testnini\1_PET.nii.gz'
img_path1 = r"E:\CKY\registration_test_20210227\chen dan yun\artery\ed.nii.gz"

subject1 = wama()
subject1.appendImageFromNifti('PET', img_path)
subject1.appendImageFromNifti('CT', img_path1)

subject1.resample('PET',aim_spacing=subject1.spacing['CT'])
writeIMG(r'D:\git\testnini\1_PETnew.nii.gz',subject1.scan['PET'],subject1.resample_spacing['PET'],subject1.origin['PET'],subject1.transfmat['PET'])


import copy
subject1.adjst_Window('CT', 321, 123)

subject1.show_bbox('CT')
coronal_min, coronal_max, sagittal_min, sagittal_max, axial_min, axial_max = subject1.getBbox('CT')

tmp_data = copy.deepcopy(subject1.scan['CT'])
tmp_data1 = copy.deepcopy(subject1.scan['CT'])
# 平滑
qwe = subject1.slice_neibor_add('CT',axis=['z'],add_num=[21],add_weights='Mean')
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square',ex_mms=24,aim_shape=[256,256,256])

subject1.makePatch(mode='slideWinND',
                   img_type='CT',
                   slices=[256//2, 256//2, 1],
                   stride=[256//2, 256//2, 40],
                   expand_r=[1, 1, 1],
                   ex_mode='square',
                   ex_mms=24,
                   aim_shape=[256, 256, 256]
                   )
reconstuct_img = slide_window_n_axis_reconstruct(subject1.patches['CT'])
patch = subject1.patches['CT']
show3D(np.concatenate([bbox_image,reconstuct_img],axis=1))






















