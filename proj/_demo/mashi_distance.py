import numpy as np
from mahalanobis import Mahalanobis
import xlrd
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
# 列表去重
import matplotlib.pyplot as plt
def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - data
    if not cov:
        cov = np.cov(x.T)
        panduan(cov)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return np.sqrt(mahal.diagonal())

class MahalanobisDistance():
    # 没有考虑的很周全的代码
    def __init__(self, X):
        """
        构造函数
        :param
            X:m * n维的np数据矩阵 每一行是一个sample 列是特征
        """
        self._PCA = None
        self._mean_x = np.mean(X, axis=0)
        mean_removed = X  # - self._mean_x
        # cov = np.dot(mean_removed.T, mean_removed) / X.shape[0] # 计算协方差矩阵
        cov = np.cov(mean_removed, rowvar=False)
        # 判断协方差矩阵是否可逆
        if np.linalg.det(cov) == 0.0:
            print('PCA ing')
            # 对数据做PCA 去掉特征值0的维度
            eig_val, eig_vec = np.linalg.eig(cov)
            e_val_index = np.argsort(-eig_val)  # 逆序排
            e_val_index = e_val_index[e_val_index > 1e-3]  # 需要特征值大于0的维度
            self._PCA = eig_vec[:, e_val_index]  # 降维矩阵 Z = XU
            PCA_X = np.dot(X, self._PCA)  # 降维
            self._mean_x = PCA_X.mean(axis=0)  # 重新计算均值 去中心
            mean_removed = PCA_X  # - self._mean_x
            # cov = np.dot(mean_removed.T, mean_removed) / PCA_X.shape[0] # 重新计算协方差矩阵
            # cov = np.cov(mean_removed, rowvar=False)
            cov = np.cov(mean_removed.T)

        panduan(cov)
        self._cov_i = np.linalg.inv(cov)  # 协方差矩阵求逆

    def __call__(self, X, y=None):
        """
        计算x与y的马氏距离 如果不传入y则计算x到样本中心的距离
        :param
            X:行向量/矩阵 样本点特征维数必须和原始数据一样
            y:行向量 样本点特征维数必须和原始数据一样
        :return
            distance 马氏距离 如果出错则返回-1
        """
        # 不考虑出错的情况 维度不符合
        if y is None:
            # 计算到样本中心的距离
            y = self._mean_x
        X_data = X.copy()

        if self._PCA is not None:
            # print(X_data.shape, self._PCA.shape,y.shape)
            X_data = np.dot(X_data, self._PCA)  # 对数据降维
            y = np.dot(np.expand_dims(y,0),self._PCA)# 也要对y降维
            y = np.squeeze(y, 0)
            # print(X_data.shape, self._PCA.shape, y.shape)

        X_data = X_data - y
        distance = np.dot(np.dot(X_data, self._cov_i), X_data.T)
        if len(X.shape) != 1:
            # X是一个矩阵
            distance = distance.diagonal()

        return np.sqrt(distance)
def list_unique(lis):
    """
    list去重复元素
    :param lis:
    :return:
    """
    return list(set(lis))
def panduan(a):
    try:
        np.linalg.inv(a)
        print('可逆')
    except:
        print('不可逆')
import warnings
warnings.filterwarnings("ignore")
#打开excel
wb = xlrd.open_workbook(r"E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\小鼠流式组_MRI分析（0208）.xlsx")
txtfile = r"E:\@data_hcc_rna_mengqi\new\mice_FCM_MRI\小鼠流式组_MRI分析（0208）-xlsx-qweqweqwe.txt"
with open(txtfile, "w") as f:
    f.write('start' + '\n')
#按工作簿定位工作表
# sh = wb.sheet_by_name('Sheet1')
sh = wb.sheet_by_name('Sheet1')
# print(sh.nrows)#有效数据行数
# print(sh.ncols)#有效数据列数
# print(sh.cell(0,0).value)#输出第一行第一列的值
# print(sh.row_values(0))#输出第一行的所有值

lines = [sh.col_values(i)[1:] for i in range(sh.ncols)]
head = [sh.col_values(i)[0] for i in range(sh.ncols)]

# 把每个样本放进字典
mice_id = list_unique(lines[0])

# 逐个样本计算
for id in mice_id:
    # 把各个层的指标拿出来，放到数组
    id_index = [i for i in range(len(lines[0])) if lines[0][i] == id]
    slices = []
    slice_names = []
    for index in id_index:
        tmp_slice = [ lines[i][index] for i in range(2,18)]
        slice_names.append(lines[2][index])
        slices.append(tmp_slice)

    # 计算马氏距离
    slices = np.array(slices)
    slice_index = slices.shape[0]//2
    # slice_index = 0
    # method1:下面这种和方法不行，具体是因为协方差矩阵不可逆，因此需要使用网上那个带有PCA的方法
    # try:
    #     from scipy.spatial.distance import pdist, squareform
    #     data_log = slices
    #     D = squareform(pdist(slices, 'mahalanobis'))
    #     print('qwe',D)
    # except:
    #     pass

    # method 2
    dis2 = None
    try:
        dis2 = mahalanobis(slices, data=slices[slice_index])
        # print(mahalanobis(slices, data=slices[slice_index]))
    except:
        pass

    madis = MahalanobisDistance(slices)
    dis = list(madis(slices, slices[slice_index]))
    dis = [abs(d) for d in dis]
    print(slice_names)
    if dis2 is not None:
        print(id, '\n','dis1@', dis, '\n', 'dis2@', dis2)
    else:
        print(id,'@',dis)

    with open(txtfile, "a") as f:
        f.write(str(slice_names)+'\n')
        f.write(str(id)+'\n')
        _ = [f.write(str(ob_an)+'\n') for ob_an in dis]







# dim = 2
# batchsize = 8
# input_2D = (np.arange(dim*batchsize) + 4*np.random.normal(0, 0.1, dim*batchsize)).reshape(batchsize,dim)
# # input_2D = (np.arange(dim*batchsize) ).reshape(dim,batchsize)
# mah1D = Mahalanobis(input_2D, 2)  # input1D[:4] is the calibration subset
# mah1D.mean = input_2D[5]
# dis = mah1D.calculate_dists(input_2D)




