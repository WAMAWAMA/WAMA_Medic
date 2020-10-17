import os
import json

sep = os.sep


def get_lines_from_txt(txt_file, encoding='utf-8'):
    with open(txt_file, "r", encoding=encoding) as f:
        try:
            data = f.read().splitlines()
            return data
        except:
            print(txt_file, 'is empty')
            pass



numerical_key = ['age', # 年龄
                 'symptom', # 症状
                 'nens_grades', # nens级别（注意要统一，WHO每年都在改,这里暂时是WHO2017标准（参考下面注释）
                 'post_recurrent', # 是否术后复发
                 'sstr2',  # 是否阳性
                 'vegfr2',# 是否阳性
                 'mgmt',# 是否阳性
                 ]


"""
WHO 2017 nens grade: 
Inzani F, Petrone G, Rindi G. 
The new World Health Organization classification for pancreatic neuroendocrine neoplasia[J]. 
Endocrinology and metabolism clinics of North America, 2018, 47(3): 463-470.
"""


def get_dict_from_meta(txt_pth):
    """
    :param txt_pth:
    :return: dict contains all metadata
    """
    lines = get_lines_from_txt(txt_pth)
    lines = [i.split('#')[0] for i in lines]
    lines = [i for i in lines if i is not '']
    obj = {}
    # 解析过程如下
    for i in lines:
        print(i)
        _keys = [_tmp.strip() for _tmp in i.split('@')[1:]]  # strip为了去除多余空格
        _level = 0 # 深入的级别


        # 检查第一级key
        if _keys[_level] not in obj.keys():
            obj[_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[_keys[_level]] = int(_keys[_level+1])
            else:
                obj[_keys[_level]] = (_keys[_level+1])
            continue
        else:
            first_key = _keys[_level]
            _level += 1


        # 检查第二级
        if _keys[_level] not in obj[first_key].keys():
            obj[first_key][_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[first_key][_keys[_level]] = int(_keys[_level+1])
            else:
                obj[first_key][_keys[_level]] = (_keys[_level+1])
            continue
        else:
            second_key = _keys[_level]
            _level += 1


        # 检查第三级
        if _keys[_level] not in obj[first_key][second_key].keys():
            obj[first_key][second_key][_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[first_key][second_key][_keys[_level]] = int(_keys[_level+1])
            else:
                obj[first_key][second_key][_keys[_level]] = (_keys[_level+1])
            continue
        else:
            third_key = _keys[_level]
            _level += 1

        # 检查第四级
        if _keys[_level] not in obj[first_key][second_key][third_key].keys():
            obj[first_key][second_key][third_key][_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[first_key][second_key][third_key][_keys[_level]] = int(_keys[_level+1])
            else:
                obj[first_key][second_key][third_key][_keys[_level]] = (_keys[_level+1])
            continue
        else:
            fouth_key = _keys[_level]
            _level += 1

        # 检查第五级
        if _keys[_level] not in obj[first_key][second_key][third_key][fouth_key].keys():
            obj[first_key][second_key][third_key][fouth_key][_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[first_key][second_key][third_key][fouth_key][_keys[_level]] = int(_keys[_level+1])
            else:
                obj[first_key][second_key][third_key][fouth_key][_keys[_level]] = (_keys[_level+1])
            continue
        else:
            fifth_key = _keys[_level]
            _level += 1

        # 检查第六级
        if _keys[_level] not in obj[first_key][second_key][third_key][fouth_key][fifth_key].keys():
            obj[first_key][second_key][third_key][fouth_key][fifth_key][_keys[_level]] = {}
        if len(_keys) == _level+2:
            if _keys[_level] in numerical_key:
                obj[first_key][second_key][third_key][fouth_key][fifth_key][_keys[_level]] = int(_keys[_level+1])
            else:
                obj[first_key][second_key][third_key][fouth_key][fifth_key][_keys[_level]] = (_keys[_level+1])
            continue
        else:
            sixth_key = _keys[_level]  # 应该不会有六级key的存在，暂时来说
            _level += 1

    return obj

if  __name__ == '__main__':
    dict_obj = get_dict_from_meta(r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\meta.txt')

    # # 可保存为json
    # if sava_as_json:
    #     json_fp = open('ob1.json', "w")
    #     json_str = json.dumps(obj, indent=4)
    #     json_fp.write(json_str)
    #     json_fp.close()




