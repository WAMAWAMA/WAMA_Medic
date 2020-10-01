from wama.utils import *



img_path = r'E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\aWITHmask4radiomics\s42_v1.nii'
mask_path = r'E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\aWITHmask4radiomics\s42_v1_m1_w.nii'

subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)


# 平滑

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