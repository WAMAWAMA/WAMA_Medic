# 去掉时间轴
from wama.utils import *
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
			if file.endswith('.' + expname) and '_' not in file:
				file_List.append(os.path.join(filepath, file))
	return file_List

path_all = r'F:\@data_guoyuan\we\TCGA-BRCA ROI\high-group'
feature_file_name = 'f'





dir_list = os.listdir(path_all)
dir_list = [path_all + sep + i for i in dir_list if '.' not in i]


fail_list = []
for index, dir in enumerate(dir_list):
	print(dir, index + 1, '/', len(dir_list))

	# savepth = dir+sep + feature_file_name + '.json'
	imgpth = dir + sep + dir.split(sep)[-1] + '-img.nii.gz'
	imgpth_cut_num = len(imgpth.split(sep)[-1])
	savepth = imgpth[:-imgpth_cut_num]+ (imgpth.split(sep)[-1]).split('.')[0]+'-norm.nii.gz'
	# savepth = img[:-12] + feature_file_name + '.json'
	if not os.path.exists(savepth):
		try:
			# print(dir)
			# 读取图像


			itkimage = sitk.ReadImage(imgpth)
			# 读取图像数据
			scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
			spacing = itkimage.GetSpacing()  # voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
			origin = itkimage.GetOrigin()  # world coordinates of origin
			transfmat = itkimage.GetDirection()  # 3D rotation matrix

			# 如果多一维，就删掉
			if len(scan.shape) == 4:
				print('cutting ', imgpth)
				scan = scan[:,:,:,0]

			cut_thre = np.percentile(scan, 99.)  # 直方图99.9%右侧值不要
			scan[scan >= cut_thre] = cut_thre


			# 覆盖储存
			itkim = sitk.GetImageFromArray(scan, isVector=False)  # 3D image
			itkim.SetSpacing(spacing)  # voxelsize
			itkim.SetOrigin(origin)  # world coordinates of origin
			itkim.SetDirection(transfmat)  # 3D rotation matrix



			sitk.WriteImage(itkim, savepth, False)






		except:
			print('@failed: ', dir)
			fail_list.append(dir)

print(fail_list)
with open(path_all + sep + 'fail.txt', "w") as f:
	_ = [f.write(str(ob_an) + '\n') for ob_an in fail_list]

