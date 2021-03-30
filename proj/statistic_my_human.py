# 统计
# 相比statistic_my.py
# 这个是分析人的，人的有个免疫评分，所以这个需要算一下组间差异（分组），参数检验什么的

import json
import os
import xlrd
import numpy as np
sep = os.sep
from scipy.stats import pearsonr
import csv
from scipy import stats
from statsmodels.stats.diagnostic import lillifors


# 正态分布测试
def check_normality(testData, printflag = False):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(testData) < 50:
        if printflag:
            print('use normal test')
        p_value = stats.normaltest(testData)
        return [p_value[0], p_value[1]]

    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(testData) < 50:
        if printflag:
            print('use shapiro test')
        p_value = stats.shapiro(testData)
        return [p_value[0], p_value[1]]

    if 300 >= len(testData) >= 50:
        if printflag:
            print('use lilliefors test')
        p_value = lillifors(testData)
        return [p_value[0], p_value[1]]

    if len(testData) > 300:
        if printflag:
            print('use kstest test')
        p_value = stats.kstest(testData, 'norm')
        return [p_value[0], p_value[1]]


def find_feature_Ttest(Data, label,use_all = True):
    # 两个都是list
    Data = np.array(Data)
    label = np.array(label)

    positiver_f = Data[label == 1]
    negative_f = Data[label == 0]
    ksresult1 = check_normality(positiver_f)
    ksresult2 = check_normality(negative_f)

    if ((ksresult1[1] > 0.05) and (ksresult2[1] > 0.05)):
        # 检验方差齐性
        leveneresult = stats.levene(positiver_f, negative_f)
        if leveneresult[1] >= 0.05:
            ttestresult = stats.ttest_ind(positiver_f, negative_f)
            static_value = ttestresult[0]
            p_value = ttestresult[1]
            method = 'ttest'
        if leveneresult[1] < 0.05:
            ttestresult = stats.ttest_ind(positiver_f, negative_f, equal_var=False)
            static_value = ttestresult[0]
            p_value = ttestresult[1]
            method = 'ttes_adj'
    elif use_all:
        nontestresult = stats.mannwhitneyu(positiver_f, negative_f)
        static_value = nontestresult[0]
        p_value = nontestresult[1]
        method = 'mannwhitneyu'

    return  [static_value, p_value, method]




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
# path_all = r'E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI'
mode_key = ['ADC','AP','HBP','T1_pos','T1_pre','t2_tra']
# mode_key = ['AP','HBP','T1_pos','T1_pre','t2_tra']

# 读取组学特征key
tmp_json = r"E:\@data_hcc_rna_huangmengqi\new\feature\human_FCM\20@T1_pos.json"
tmp_json_dict = load_json(tmp_json)
Radiomics_f_key = list(tmp_json_dict.keys())

# 读取流式特征
wb = xlrd.open_workbook(r"E:\@data_hcc_rna_huangmengqi\new\human_FCM\03 Tu_Immune.xlsx")
sh = wb.sheet_by_name('03 Tu_Immune')
lines = [sh.col_values(i)[1:] for i in range(sh.ncols)]
head = [sh.col_values(i)[0] for i in range(sh.ncols)]
FCM_f_key = head[1:-1]



# 做统计
csv_file = r'E:\@data_hcc_rna_huangmengqi\new\human_FCM\statics.csv'
f = open(csv_file, 'w', encoding='utf-8',newline='')
f.close()

csv_line = []
failed_list = []
json_root = r'E:\@data_hcc_rna_huangmengqi\new\feature\human_FCM'
for index1, mode in enumerate(mode_key):
    for index2, Radiomics_f_key_ in enumerate(Radiomics_f_key):
        print('@',index1,'/',len(mode_key), '@',index2, '/', len(Radiomics_f_key))
        json_path = [json_root + sep + lines[0][i][2:] + '@' + mode + '.json'
                     for i in range(len(lines[0]))]

        # 获取组学特征
        radiomics_f = [get_radiomics_f(Radiomics_f_key_, json_file) for json_file in json_path]
        # 缺失值用均值补齐
        if None in radiomics_f:
            print('缺失：',mode)
            radiomics_f_mean = np.mean([i for i in radiomics_f if i is not None])
            for index, i in enumerate(radiomics_f):
                if i is None:
                    radiomics_f[index] = radiomics_f_mean


        # 获取流式特征并做person相关
        tmp_csv_line =mode+','+Radiomics_f_key_
        for FCM_f_key_ in FCM_f_key:
            fcm_index = head.index(FCM_f_key_)
            FCM_f = lines[fcm_index]

            # 做person相关
            r, p = pearsonr(radiomics_f, FCM_f)

            tmp_csv_line = tmp_csv_line+','+str(r)+','+str(p)

        # 获取免疫分数，并做参数或非参数检验
        immune_label = lines[-1]
        static_value, p_value, method = find_feature_Ttest(radiomics_f, immune_label)
        tmp_csv_line = tmp_csv_line + ',' + str(static_value) + ',' + str(p_value) + ',' + str(method)


        f = open(csv_file, 'a', encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(list(tmp_csv_line.split(',')))
        f.close()












