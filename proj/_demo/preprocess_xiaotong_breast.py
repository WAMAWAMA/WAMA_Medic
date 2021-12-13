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
			if file.endswith('.' + expname) and '_' not in file:
				file_List.append(os.path.join(filepath, file))
	return file_List


def resave(savepth, imgpth):
	itkimage = sitk.ReadImage(imgpth)
	# 读取图像数据
	scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
	spacing = itkimage.GetSpacing()  # voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
	origin = itkimage.GetOrigin()  # world coordinates of origin
	transfmat = itkimage.GetDirection()  # 3D rotation matrix

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
	path_all = r'E:\task\BC_data\raw_data\test1_sfy\sfy_right_data\sfy_right_data\low\low'
	save_pth = r'E:\task\BC_data\raw_data_nii\test_sfy\low'
	dir_list = os.listdir(path_all)
	dir_list = [path_all+sep+i for i in dir_list if '.' not in i]
	dir_list_final = []
	for dir in dir_list:
		tmp_dir_list = os.listdir(dir)
		tmp_dir_list = [dir+sep+i for i in tmp_dir_list if 'C3' in i]
		for i in tmp_dir_list:
			dir_list_final.append(i)
	print(len(dir_list_final))


	fail_list = []
	for index, dirr in enumerate(dir_list_final):
		print(index+1,'/',len(dir_list_final),dirr)
		if get_filelist_frompath4newnii(dirr, 'gipl'):
			_tmp_dir = dirr
		else:
			_tmp_dir =dirr+sep+ [i for i in os.listdir(dirr) if os.path.isdir(dirr+sep+i)][0]
		# 再判断一层
		if not get_filelist_frompath4newnii(_tmp_dir, 'gipl'):
			_tmp_dir = _tmp_dir + sep + [i for i in os.listdir(_tmp_dir) if os.path.isdir(_tmp_dir + sep + i)][0]
		if not get_filelist_frompath4newnii(_tmp_dir, 'gipl'):
			_tmp_dir = _tmp_dir + sep + [i for i in os.listdir(_tmp_dir) if os.path.isdir(_tmp_dir + sep + i)][0]

		#
		for nii in get_filelist_frompath4newnii(_tmp_dir,'gipl'):
			try:
				if not os.path.exists(save_pth+sep+nii.split(sep)[-1][:-5]+'.nii.gz'):
					resave(save_pth+sep+nii.split(sep)[-1][:-5]+'.nii.gz', nii)
					print('scceed')
			except:
				print(nii,'failed' )
				fail_list.append(nii)




	with open(path_all+sep+'f_list.txt', "w") as f:
		_ = [f.write(i + '\n')for  i in fail_list]






