# 去掉时间轴
# from wama.utils import *
import numpy as np
import SimpleITK as sitk
import os
sep = os.sep
import json
import warnings
warnings.filterwarnings("ignore")
def get_filelist_frompath4newnii(filepath, expname, sample_id=None):
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
			if file.endswith('.' + expname):
				file_List.append(os.path.join(filepath, file))
	return file_List


def resave(savepth, imgpth):
	itkimage = sitk.ReadImage(imgpth)
	# 读取图像数据
	scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
	spacing = itkimage.GetSpacing()  # voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
	origin = itkimage.GetOrigin()  # world coordinates of origin
	transfmat = itkimage.GetDirection()  # 3D rotation matrix

	scan[scan>=1] = 1
	# 覆盖储存
	itkim = sitk.GetImageFromArray(scan, isVector=False)  # 3D image
	itkim.SetSpacing(spacing)  # voxelsize
	itkim.SetOrigin(origin)  # world coordinates of origin
	itkim.SetDirection(transfmat)  # 3D rotation matrix

	sitk.WriteImage(itkim, savepth, False)

# tmp_s = []
# tmp_s +=dir_list_final
# tmp_s = [i.split(sep)[-2] for i in tmp_s]
# len(list_unique(tmp_s))
#
#
# listss = get_filelist_frompath4newnii(r'E:\task\BC_data\raw_data_nii\train\low', 'nii.gz')
# listss = [i.split(sep)[-1][:-7] for i in listss if 'seg' not in i]
# for i in tmp_s:
# 	if i not in listss:
# 		print(i)

# tmp_s.sort()

if __name__ == '__main__':
	path_all = r'E:\9database\database\@private_IBD_CTX\data\valid\roi'
	save_pth = r'E:\9database\database\@private_IBD_CTX\data\valid\roi_new'

	dir_list_final = get_filelist_frompath4newnii(path_all, 'nii.gz')
	for index, dirr in enumerate(dir_list_final):
		resave(save_pth + sep + dirr.split(sep)[-1], dirr)








