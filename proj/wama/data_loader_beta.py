from torch.utils import data


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
        else:  # 如果为None，则有传递（or重新命名)的作用
            _tmp_dict = {}
            for _index, _key in enumerate(method['out_key']):
                _tmp_dict[_key] = tmp_dict[method['in_key'][_index]]
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
        # print('checking method: t method ',str(method['method']),
        #       '  t+1 method ', str(next_method['method']))
        if method['out_key'] != next_method['in_key']:
            raise ValueError('out_keys do not match in_key')
    # 其次检查None方法是否in和out数量一致
    for method in pipeline:
        if method['method'] is None:
            if len(method['in_key']) != len(method['out_key']):
                raise ValueError('None_method must have the same length of in_key and out_keys')
    return 1


class wama_dataset(data.Dataset):
    def __init__(self, input_dict_list, pipeline_list):
        """
        :param input_dict_list: 是一个list，每个element是一个sample
        :param pipeline_list: 由许多个pipeline构成的list，会依次执行其中的pipeline
        :param mode:
        """

        self.input_dict_list = input_dict_list
        self.pipeline_list = pipeline_list

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



def get_loader(input_dict_list, pipeline_list, num_workers = 0, pin_memory=False, batch_size = 3, drop_last = True):
    dataset = wama_dataset(input_dict_list=input_dict_list, pipeline_list=pipeline_list)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  pin_memory=pin_memory)
    return data_loader
