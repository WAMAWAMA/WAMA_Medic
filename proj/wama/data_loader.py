# 先看看，能否集成torch的，分为cache和普通两个版本，cache的会提前读进来
from torch.utils import data
import random
import torch
import numpy as np
from wama.utils import load_from_pkl



"""扩增操作的两个顺序
1）在dataloader中__getitem__对每个单独的样本进行扩增，最后组合为batch，这样可能会慢，但是会保证pin_memory没问题？
2）在dataloader外，每次得到一个batch后，将tensor形式的batch转换为numpy进行扩增，之后再从numpy转为tensor，不过这样的话pin_memory可能会有问题
"""



# 先构建一个数组再说，然后把patch都搞进来
dataset = patches

class wama_dataset(data.Dataset):
    """
    使用这个dataset，扩增会在形成batch前进行
    """
    def __init__(self, patches_path_list, mode='train', augmenter = None):
        """
        :param patches_path_list: path list of wama.utils.patch_tmp objects （patch_tmp 对象的完整路径的列表）
        :param mode: 'train' or sth. , if 'train', do augmentation
        :param augmenter: transform after compose by 'batchgenerators', refer to 'https://github.com/MIC-DKFZ/batchgenerators'
        """
        self.patches_path_list = patches_path_list
        self.mode = mode
        self.augmenter = augmenter


    def __getitem__(self, indexx):
        # 读取patch
        tmp_patch = load_from_pkl(self.patches_path_list[indexx])

        # 取出单个patch的数据
        tmp_patch_data = tmp_patch.data
        tmp_patch_mask = tmp_patch.mask
        tmp_patch_info = tmp_patch.info

        if self.mode == 'train' and self.augmenter is not None:
            # 由于batchgenerators处理 5维（3D） or  4维（2D）数据，so需要增加两个axis
            tmp_patch_data = np.expand_dims(np.expand_dims(tmp_patch_data, axis=0),axis=0)
            if tmp_patch_mask is not None:
                tmp_patch_mask = np.expand_dims(np.expand_dims(tmp_patch_mask, axis=0),axis=0)

            # 将data和mask放入字典以备扩增
            if tmp_patch_mask is not None:
                tmp_patch_dict = {'data': tmp_patch_data, 'seg':tmp_patch_mask}
            else:
                tmp_patch_dict = {'data': tmp_patch_data}

            # 扩增
            tmp_patch_dict = self.augmenter(tmp_patch_dict)
            print('扩增')

            # 扩增后需要将维度还原，即去掉0，1维度
            tmp_patch_data = np.squeeze(np.squeeze(tmp_patch_dict['data'], axis=0),axis=0)
            if tmp_patch_mask is not None:
                tmp_patch_mask = np.squeeze(np.squeeze(tmp_patch_dict['seg'], axis=0),axis=0)

        # 将data、mask、info全部放到字典里，返回sample
        return_dict = {}
        return_dict['data'] = tmp_patch_data
        if tmp_patch_mask is not None:
            return_dict['seg'] = tmp_patch_mask
        for k in tmp_patch_info.keys():
            return_dict['info'+k] = str(tmp_patch_info[k])
        return return_dict

    def __len__(self):
        return len(self.patches_path_list)

class wama_dataset_cache(data.Dataset):
    """
    使用这个dataset，扩增会在形成batch前进行
    """
    def __init__(self, patches, mode='train', augmenter = None):
        """
        :param patches: list of wama.utils.patch_tmp objects （patch_tmp对象的列表）
        :param mode: 'train' or sth. , if 'train', do augmentation
        :param augmenter: transform after compose by 'batchgenerators', refer to 'https://github.com/MIC-DKFZ/batchgenerators'
        """
        self.patches = patches
        self.mode = mode
        self.augmenter = augmenter


    def __getitem__(self, indexx):
        # 取出单个patch的数据
        tmp_patch = self.patches[indexx]
        tmp_patch_data = tmp_patch.data
        tmp_patch_mask = tmp_patch.mask
        tmp_patch_info = tmp_patch.info

        if self.mode == 'train' and self.augmenter is not None:
            # 由于batchgenerators处理 5维（3D） or  4维（2D）数据，so需要增加两个axis
            tmp_patch_data = np.expand_dims(np.expand_dims(tmp_patch_data, axis=0),axis=0)
            if tmp_patch_mask is not None:
                tmp_patch_mask = np.expand_dims(np.expand_dims(tmp_patch_mask, axis=0),axis=0)

            # 将data和mask放入字典以备扩增
            if tmp_patch_mask is not None:
                tmp_patch_dict = {'data': tmp_patch_data, 'seg':tmp_patch_mask}
            else:
                tmp_patch_dict = {'data': tmp_patch_data}

            # 扩增
            tmp_patch_dict = self.augmenter(tmp_patch_dict)
            print('扩增')

            # 扩增后需要将维度还原，即去掉0，1维度
            tmp_patch_data = np.squeeze(np.squeeze(tmp_patch_dict['data'], axis=0),axis=0)
            if tmp_patch_mask is not None:
                tmp_patch_mask = np.squeeze(np.squeeze(tmp_patch_dict['seg'], axis=0),axis=0)

        # 将data、mask、info全部放到字典里，返回sample
        # return torch.Tensor(tmp_patch.data)  # 直接按照tensor形式返回
        # return tmp_patch.data  # 会被torch强制转换为tensor
        # return [tmp_patch.data]  # 如果batch是list，torch会遍历list，并将里面的numpy转换为tensor
        # return {'data':tmp_patch.data,'otherstr':'str','othernum':2}  # 如果是字典，那就遍历字典，numpy和number都会被转换为tensor，str会被转换成list
        return_dict = {}
        return_dict['data'] = tmp_patch_data
        if tmp_patch_mask is not None:
            return_dict['seg'] = tmp_patch_mask
        for k in tmp_patch_info.keys():
            return_dict['info'+k] = str(tmp_patch_info[k])
        return return_dict

    def __len__(self):
        return len(self.patches)



def get_loader(patches, num_workers = 0, mode='train', preload = True, augmenter = None, pin_memory=False, batch_size = 3):
    if preload:
        dataset = wama_dataset_cache(patches=patches, mode=mode, augmenter = augmenter)
    else:
        dataset = wama_dataset(patches_path_list=patches, mode=mode,augmenter=augmenter )


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=pin_memory)
    return data_loader



for i, sample in enumerate(data_loader):
    print('load ',i+1)
    tmp_sample = sample
