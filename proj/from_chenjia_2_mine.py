# 把嘉哥预处理的数据，转换为自己的格式（dataset，PKL那个格式）
import numpy as np
import os
import h5py
import pickle
sep = os.sep
from wama.utils import show2D

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



def save_as_pkl(save_path, obj):
	data_output = open(save_path, 'wb')
	pickle.dump(obj, data_output)
	data_output.close()



path_all = r'E:\CK19\out'
# dataset_name = r'HCC_MVI_CK19_inter'
dataset_name = r'HCC_MVI_CK19_exter'

case_dir = os.listdir(path_all)

train_set = []
for index, dir in enumerate(case_dir):
	print(index+1,'/',len(case_dir))
	h5_list = get_filelist_frompath4newnii(path_all+sep+dir, 'h5')
	for h5_file in h5_list:
		data = h5py.File(h5_file, 'r')
		case = dict(
		img = (data['img'].value).astype(np.float32),
		mask = (data['gt'].value).astype(np.int8),
		label_ck19 = data['ck19'].value,
		label_mvi = data['mvi'].value,
		img_path = dir,
		mask_path = dir,
		slice_index = int((h5_file.split(sep)[-1]).split('.')[0]),
		)
		data.close()
		train_set.append(case)

dataset = dict(
	train_set = train_set,
	train_set_num = len(train_set),
	dataset_name = dataset_name,
)
# show2D(train_set[12]['image'])
# show2D(train_set[12]['mask'])

save_as_pkl(path_all+sep+r'dataset_chenjia.pkl',
                [dataset, 'None_config'])
















