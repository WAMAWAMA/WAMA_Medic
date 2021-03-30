import os
sep = os.sep

txt_file1 = r'D:\新整理经过手术的病人.txt'   # excel复制到txt里即可
txt_file2 = r'D:\旧宋整理的全部样本.txt'

def list_unique(lis):
    """
    list去重复元素
    :param lis:
    :return:
    """
    return list(set(lis))

def get_list_from_txt(txt_file):
    """
    读取txt，返回list（int）
    :param txt_file:
    :return:
    """
    with open(txt_file, "r", encoding='utf-8') as f:
        data = f.read().splitlines()
    # data = [int(i)for i in data]  # 转换为int
    # 去除空格，不过不用了，已经int了
    data = [i.strip() for i in data]
    # _tmp.strip()
    return data

def get_set(setA, setB):
    # 找出交集
    inAB = list(set(setA).intersection(set(setB)))
    print(len(inAB))
    print(len(setA))
    print(len(list_unique(setA)))
    print(len(list_unique(setB)))
    print(len(setB))
    # 找出A - 交集
    inAnotinAB = list(set(setA).difference(set(inAB)))
    print(len(inAnotinAB))
    # 找出B - 交集
    inBnotinAB = list(set(setB).difference(set(inAB)))
    print(len(inBnotinAB))

    return [inAB,inAnotinAB,inBnotinAB]


setA = list_unique(get_list_from_txt(txt_file1))
setB = list_unique(get_list_from_txt(txt_file2))
inAB,inAnotinAB,inBnotinAB = get_set(setA, setB)
