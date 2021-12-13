from wama.utils import *

# img_path = r"E:\@data_hcc_rna_mengqi\new\human_yuhou\P018\T2WI\nifti.nii.gz"
# # img_path = r'E:\@data_hcc_rna_mengqi\new\human_FCM\05\Tu05.nii.gz'
img_path1 = r"D:\git\testnini\s1_v1_m1_w.nii"
#
subject1 = wama()
subject1.appendImageFromNifti('PET', img_path1)
#
subject1.resample('PET',aim_spacing=[3.,3.,3.], order=0)
show3D(subject1.scan['PET'])
# writeIMG(r"E:\@data_hcc_rna_mengqi\new\human_yuhou\P018\T2WI\nifti_111.nii.gz",subject1.scan['PET'],subject1.resample_spacing['PET'],subject1.origin['PET'],subject1.transfmat['PET'])
#










# todo
def resampling(file_path, only_return_spacing = False):
    img_path = file_path+ sep+'nifti.nii.gz'
    subject1 = wama()
    subject1.appendImageFromNifti('PET', img_path)

    if only_return_spacing:
        return subject1.spacing['PET']

    subject1.resample('PET', aim_spacing=[1., 1., 1.], order=3)
    writeIMG(file_path+ sep+'nifti_111.nii.gz',
             subject1.scan['PET'], subject1.resample_spacing['PET'], subject1.origin['PET'], subject1.transfmat['PET'])


# subject1.scan['PET'] = subject1.scan['PET'][:,:,::-1]
# writeIMG(r'E:\@data_hcc_rna_mengqi\new\human_FCM\16\Tu17_111.nii.gz',subject1.scan['PET'],subject1.spacing['PET'],subject1.origin['PET'],subject1.transfmat['PET'])


# todo
"""
"""
import SimpleITK as sitk
import os

sep = os.sep



path_all = r'E:\@data_hcc_rna_mengqi\new\human_yuhou'
dir_list = os.listdir(path_all)
dir_list = [path_all + sep + i for i in dir_list if '.' not in i]
dir_list_final = []
for dir in dir_list:
    tmp_dir_list = os.listdir(dir)
    tmp_dir_list = [dir + sep + i for i in tmp_dir_list if '.' not in i]
    for i in tmp_dir_list:
        dir_list_final.append(i)

fail_list = []
spacing_list = []
for index, i in enumerate(dir_list_final):
    print(i, index + 1, '/', len(dir_list_final))
    if os.path.exists(i + sep + 'nifti.nii.gz'):
        try:
            resampling(i)
            # spacing_list.append(resampling(i, only_return_spacing=True))
        except:
            fail_list.append(i)


# dim1_spacing = [i[0] for i in spacing_list]
# dim2_spacing = [i[1] for i in spacing_list]
# dim3_spacing = [i[1] for i in spacing_list]
#
# import matplotlib.pyplot as plt
# plt.subplot(1,3,1)
# plt.hist(dim1_spacing,color='r')
# plt.subplot(1,3,2)
# plt.hist(dim2_spacing,color='g')
# plt.subplot(1,3,3)
# plt.hist(dim3_spacing,color='b')
# plt.show()

