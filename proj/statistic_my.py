# 统计
import json
import os
import xlrd
import numpy as np
sep = os.sep
from scipy.stats import pearsonr
import csv





def load_json(json_path):
    with open(json_path, 'r') as load_f:
        return json.load(load_f)

def list_unique(lis):
    # 列表去重
    return list(set(lis))

def get_radiomics_f(key, json_path):
    try:
        json_dict = load_json(json_path)
        return json_dict[key]
    except:
        return None

# 获取序列名称(mode key)
path_all = r'E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI'
dir_list = os.listdir(path_all)
dir_list = [path_all + sep + i for i in dir_list if '.' not in i]
dir_list_final = []
for dir in dir_list:
    sample_dict = {}
    tmp_dir_list = os.listdir(dir)
    tmp_dir_list = [ i for i in tmp_dir_list if '.' not in i]
    for i in tmp_dir_list:
        dir_list_final.append(i)

dir_list_final = list_unique(dir_list_final)
mode_key = [i for i in dir_list_final if 'Series' not in i]

# 读取组学特征key
tmp_json = r"E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\Z151\grasp@Z151_slice4.json"
tmp_json_dict = load_json(tmp_json)
Radiomics_f_key = list(tmp_json_dict.keys())

# 读取流式特征
wb = xlrd.open_workbook(r"E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\小鼠HCC模型流式分析结果记录.xlsx")
sh = wb.sheet_by_name('Sheet1')
lines = [sh.col_values(i)[1:] for i in range(sh.ncols)]
head = [sh.col_values(i)[0] for i in range(sh.ncols)]
FCM_f_key = head[3:-1]



# 做统计
f = open(r'E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\statics.csv', 'w', encoding='utf-8',newline='')
f.close()

csv_line = []
failed_list = []
json_root = r'E:\@data_hcc_rna_mengqi\new\feature\mice_FCM_MRI'
for index1, mode in enumerate(mode_key):
    for index2, Radiomics_f_key_ in enumerate(Radiomics_f_key):
        print('@',index1,'/',len(mode_key), '@',index2, '/', len(Radiomics_f_key))
        json_path = [json_root + sep + lines[0][i] + sep + mode + '@' + lines[0][i] + '_slice' + str(int(lines[2][i]))+ '.json'
                     for i in range(len(lines[0]))]
        for i in range(len(json_path)):
            if '0406HBP' in json_path[i]:
                json_path[i] = json_root + sep + lines[0][i] + sep + mode + '@' + lines[0][i] + '@Unenhanced_slice' + str(int(lines[2][i]))+ '.json'

        # 获取组学特征
        radiomics_f = [get_radiomics_f(Radiomics_f_key_, json_file) for json_file in json_path]
        # 缺失值用均值补齐
        if None in radiomics_f:
            print('缺失：',mode)
            radiomics_f_mean = np.mean([i for i in radiomics_f if i is not None])
            for index, i in enumerate(radiomics_f):
                if i is None:
                    radiomics_f[index] = radiomics_f_mean


        # 获取流式特征
        tmp_csv_line =mode+','+Radiomics_f_key_
        for FCM_f_key_ in FCM_f_key:
            fcm_index = head.index(FCM_f_key_)
            FCM_f = lines[fcm_index]

            # 做person相关
            r, p = pearsonr(radiomics_f, FCM_f)

            tmp_csv_line = tmp_csv_line+','+str(r)+','+str(p)

        f = open(r'E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\statics.csv', 'a', encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(list(tmp_csv_line.split(',')))
        f.close()












