# 基于pipeline的dataloader
# from
from torch.utils import data
import random
import torch
import numpy as np
from wama.utils import load_from_pkl
from scipy.ndimage import zoom

def readimg(file_name,file_path,out_size):
    print('readimg file:',file_name, 'at:',file_path, "@size ",out_size)
    return {'img':np.zeros([50, 50]),'file_name': file_name}

def imgaug(img, file_name):
    print('imgaug @ shape:',img.shape)
    return {'img': img, 'file_name': file_name}



"""






]

pipeline_list
其中每个element都对应一个pipeline(即method的list），
调用会按照pipeline里面的method顺序来调用method，最后method的输出会存入输出字典out_dict

每个method都是个字典，字典键包括，'method','in_key','out_key'
通过这样的规则，使得method可以串起来,但是要记住，有些in_key如果想要传递到最后，就要不断传递，详见下面例子中的file_name

"""

# pipeline是有method组成的，一个method即为pipeline中的一个step
# 注意in_key和out_key要匹配，顺序也要
pipeline1 = [
{'method': readimg, 'in_key':['file_name','file_path','out_size'],'out_key':['img','file_name']},
{'method': imgaug,  'in_key':['img','file_name'],                   'out_key':['img','file_name']} ]

pipeline2 = [
{'method': None, 'in_key':['img'], 'out_key':['img']},
{'method': None, 'in_key':['img'], 'out_key':['img']},
{'method': None, 'in_key':['img'], 'out_key':['img']}]
# 如果method为None，则直接传递变量给下一个函数or字典

pipeline3 = [
{'method': None, 'in_key':['mask'], 'out_key':['mask']},
{'method': None, 'in_key':['mask'], 'out_key':['mask']},
{'method': None, 'in_key':['mask'], 'out_key':['mask']},
{'method': None, 'in_key':['mask'], 'out_key':['mask']},
{'method': None, 'in_key':['mask'], 'out_key':['mask']},
{'method': None, 'in_key':['mask'], 'out_key':['mask']}]
# 如果method为None，则直接传递变量给下一个函数or字典

pipeline_list = [pipeline1,
                 pipeline2,
                 pipeline3]


"""

"""

def run_pipeline(in_dict, pipeline):
    """
    按照顺序执行pipeline中各个方法
    :param in_dict: 一个dict，负责储存pipeline的输入(包含第一个method的输入key即可）
    :param pipeline: 要执行的pipeline
    :return: 一个dict，即pipeline中最后一个method的输出
    """
    # 取出第一个method的in_key的东西，放到一个新的dict
    tmp_dict = {}
    for _key in pipeline[0]['in_key']:
        tmp_dict[_key] = in_dict[_key]
    for method in pipeline:
        if method['method'] is not None:
            tmp_dict = method['method'](**tmp_dict)
        else:  # 如果为None，则有传递（or过滤）的作用
            _tmp_dict = {}
            for _key in method['in_key']:
                _tmp_dict[_key] = tmp_dict[_key]
            tmp_dict = _tmp_dict
    return tmp_dict

def chechout_pipeline(pipeline):
    """
    检查pipeline中各个方法的一致性,即输入输出能否串联上，
    如果方法为 None，则只是传递作用，对应的in_key和out_key必须一致
    :param pipeline:
    :return:
    """
    step_num = len(pipeline)
    # 首先检查输入输出能否串联上
    for index in range(step_num-1):
        method = pipeline[index]
        next_method = pipeline[index+1]
        print('checking method: t method ',str(method['method']),
              '  t+1 method ', str(next_method['method']))
        if method['out_key'] != next_method['in_key']:
            raise ValueError('out_keys do not match in_key')
    # 其次检查None方法是否in和out一致
    for method in pipeline:
        if method['method'] is None:
            if method['in_key'] != method['out_key']:
                raise ValueError('None_method must have the same in_key and out_keys')
    return 1

chechout_pipeline(pipeline1)
chechout_pipeline(pipeline2)
chechout_pipeline(pipeline3)


indict = {'file_name':'qwe.png',
          'file_path':r'root',
          'out_size':256,
          'sthelse':123123,
          'img':np.zeros([3, 3, 3]),
          'mask':np.ones([4, 4, 4])}

outdict1 = run_pipeline(in_dict=indict,pipeline=pipeline1)
outdict2 = run_pipeline(in_dict=indict,pipeline=pipeline2)
outdict3 = run_pipeline(in_dict=indict,pipeline=pipeline3)







"""
data_list
其中每个data都是一个字典，利用键值关系储存数据
例子：
input_dict_list = [
# 第一个样本
{'img':np.zeros([128,128,3]), 'mask':np.ones([128,128]), 
 'file_name':'000012212.pkl',
 'value_age':19, 'value_hight':180,
 'cate_grade':1, 'cate_sex':1 }
# 第二个样本
{'img':np.zeros([128,128,3]), 'mask':np.ones([128,128]), 
 'file_name':'000012222.pkl',
 'value_age':19, 'value_hight':180,
 'cate_grade':2, 'cate_sex':0 }
"""

class wama_dataset(data.Dataset):
    def __init__(self, input_dict_list, pipeline_list, mode):
        """
        :param input_dict_list: 是一个list，每个element是一个sample
        :param pipeline_list: 由许多个pipeline构成的list，会依次执行其中的pipeline
        :param mode:
        """

        self.input_dict_list = input_dict_list
        self.pipeline_list = pipeline_list
        self.mode = mode

    def __len__(self):
        return len(self.input_dict_list)

    def __getitem__(self, index):
        # 取出一个sample
        indict = self.input_dict_list[index]
        # 提前构造返回值，也是个dict结构
        out_dict = {}
        # 一次调用pipeline_list中的各个pipeline
        for pipeline in self.pipeline_list:
            # 检查pipeline
            chechout_pipeline(pipeline)
            # 执行pipeline
            tmp_dict = run_pipeline(in_dict=indict, pipeline=pipeline)
            # 储存结果（or覆盖之前pipeline的某些结果）
            out_dict.update(tmp_dict)

        # 注意，out_dict中每一个值只能为‘字符串’，值和数组（torch限制），注意自查
        return out_dict




