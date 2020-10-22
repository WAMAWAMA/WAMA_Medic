from wama.utils import *
import copy


img_path = r'D:\git\testnini\s22_v1.nii'
mask_path = r'D:\git\testnini\s22_v1_m1.nii'

subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)

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





















