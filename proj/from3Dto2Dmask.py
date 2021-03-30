# 把3D的mask沿着z轴拆开，giao
from wama.utils import *
import numpy as np
import copy
import os
sep = os.sep

def get_filelist_frompath(filepath, expname, sample_id=None):
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('_')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname) and '_' not in file:
                file_List.append(os.path.join(filepath, file))
    return file_List

def nii3Dto2D(img_path):
    # img_path = r"E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\0406\0406HBP.nii.gz"
    # img_path = r'E:\@data_hcc_rna_mengqi\new\human_FCM\05\Tu05.nii.gz'
    # img_path1 = r'D:\git\testnini\1_CT.nii.gz'


    subject1 = wama()
    subject1.appendImageFromNifti('PET', img_path)
    scan = subject1.scan['PET']
    slice_index = [i for i in range(scan.shape[2]) if np.sum(scan[:,:,i] != 0)]
    for i in slice_index:
        tmp_scan = copy.deepcopy(scan)
        for ii in slice_index: # 把其他层置0
            if ii != i:
                tmp_scan[:,:,ii] = tmp_scan[:,:,ii] * 0.
        save_pth = img_path[:-7]+'_slice'+str(i) + '.nii.gz'
        writeIMG(save_pth,tmp_scan,subject1.spacing['PET'],subject1.origin['PET'],subject1.transfmat['PET'])


path_all = r'E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI'
dir_list = os.listdir(path_all)
dir_list = [path_all + sep + i for i in dir_list if '.' not in i]

nii_list = []
for dirr in dir_list:
    tmp_list = get_filelist_frompath(dirr, 'nii.gz')
    for i in tmp_list:
        nii_list.append(i)

for niifile in nii_list:
    try:
        nii3Dto2D(niifile)
    except:
        print(niifile, 'failed')
















