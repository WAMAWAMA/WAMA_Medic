import numpy as np
import SimpleITK as sitk
import warnings
from scipy.ndimage import zoom
import pickle
from mayavi import mlab
import os
import math
import matplotlib.pyplot as plt
from copy import deepcopy

# 某些库不是必要的，so，若有则导入，否则不需要导入
try:
    from mayavi import mlab
    # from qweqwe import mlab
    print ('mayavi already imported')
    mayavi_exist_flag = True
except:
    print ('no mayavi')
    mayavi_exist_flag = 0



class patch_tmp():
    """
    patch类，只是作为存放数据的容器
    """
    def __init__(self):
        self.data = None
        self.mask = None
        self.info = {}


def save_as_pkl(save_path, obj):
    data_output = open(save_path, 'wb')
    pickle.dump(obj, data_output)
    data_output.close()

def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data



def resize3D(img, aimsize, order = 2):
    """

    :param img: 3D array
    :param aimsize: list, one or three elements, like [256], or [256,56,56]
    :return:
    """
    _shape =img.shape
    if len(aimsize)==1:
        aimsize = [aimsize[0] for _ in range(3)]
    return zoom(img, (aimsize[0] / _shape[0], aimsize[1] / _shape[1], aimsize[2] / _shape[2]), order=order)  # resample for cube_size

def show1D(vector):
    plt.plot(vector)
    plt.show()


def show2D(img2D):
    plt.imshow(img2D,cmap=plt.cm.gray)
    plt.show()

def show3D(img3D):
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
    mlab.show()

def show3Dslice(image):
    """
    查看3D体，切片模式
    :param image:
    :return:
    """
    mlab.volume_slice(image, colormap='gray',
                       plane_orientation='x_axes', slice_index=10)         # 设定x轴切面
    mlab.volume_slice(image, colormap='gray',
                       plane_orientation='y_axes', slice_index=10)         # 设定y轴切面
    mlab.volume_slice(image, colormap='gray',
                      plane_orientation='z_axes', slice_index=10)          # 设定z轴切面
    mlab.show()

def connected_domain_3D(image):
    """
    返回3D最大连通域
    :param image: 二值图
    :return:
    """
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    # for idx, i in enumerate(num_list_sorted[1:]):
    #     if area_list[i-1] >= (largest_area//10):
    #         final_label_list.append(i)
    #     else:
    #         break
    output = sitk.GetArrayFromImage(output_ex)

    output = output==final_label_list
    output = output.astype(np.float32)
    # for one_label in num_list:
    #     if  one_label in final_label_list:
    #         continue
    #     x, y, z, w, h, d = stats.GetBoundingBox(one_label)
    #     one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
    #     output[z: z + d, y: y + h, x: x + w] *= one_mask
    #
    # if mask:
    #     output = (output > 0).astype(np.uint8)
    # else:
    #     output = ((output > 0)*255.).astype(np.uint8)
    # output = output.astype(np.float32)
    # output[output == final_label_list] = -1.
    # output = output < 0.1
    # output = output.astype(np.uint8)
    return output

def mat2gray(image):
    """
    归一化函数
    :param image: ndarray
    :return:
    """
    # as dtype = np.float32
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image)-np.min(image)+ 1e-14)
    return image

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / (sigma+ 1e-14)

def getImgWorldTransfMats(spacing, transfmat):
    """
    用不到的函数，仅供科普
    :param spacing:
    :param transfmat:
    :return:
    """
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3], transfmat[3:6], transfmat[6:9]])
    for d in range(3):
        transfmat[0:3, d] = transfmat[0:3, d] * spacing[d]

    # image to world coordinates conversion matrix  真实世界的基向量
    transfmat_toworld = transfmat

    # world to image coordinates conversion matrix 取倒数操作，
    # 这样从真实世界变回image坐标系，就不用除以真实世界的基向量了，直接乘这个就行
    transfmat_toimg = np.linalg.inv(transfmat)

    return transfmat_toimg, transfmat_toworld

def adjustWindow(img, WW, WL):
    """
    调整窗宽窗位的函数
    :param img:
    :param WW: 窗宽
    :param WL: 窗位
    :return:
    """
    img[img>WL+WW*0.5] = WL+WW*0.5
    img[img<WL-WW*0.5] = WL-WW*0.5
    return img

# todo
# 要好好确认下spacing对应的维度，以及scan对应的维度
def readIMG(filename):
    """
    read mhd/NIFTI image
    :param filename:
    :return:
    scan 图像，ndarray，注意这里已经改变了axis，返回的图axis对应[coronal,sagittal,axial], [x,y,z]
    spacing：voxelsize，对应[coronal,sagittal,axial], [x,y,z]
    origin：realworld 的origin
    transfmat：方向向量组成的矩阵，一组基向量，3D的话，一般是(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)，也即代表
                [1,0,0],[0,1,0],[0,0,1]三个基向量，分别对应
    """
    itkimage = sitk.ReadImage(filename)
    # 读取图像数据
    scan = sitk.GetArrayFromImage(itkimage) #3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
    scan = np.transpose(scan, (1,2,0))     # 改变axis，对应的axis[coronal,sagittal,axial]，即[y，x，z]
    # 读取图像信息
    spacing = itkimage.GetSpacing()        #voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z] ？？？ 有待确认
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    axesOrder = ['coronal', 'sagittal', 'axial']  # 调整顺序可以直接axesOrder = [axesOrder[0],axesOrder[2],axesOrder[1]]
    return scan,spacing,origin,transfmat,axesOrder

def writeIMG(filename,scan,spacing,origin,transfmat):
    """
    :param filename:
    :param scan: 注意这个scan的axis必须为[axial,coronal,sagittal],也即[z，y，x]
    :param spacing:
    :param origin:
    :param transfmat:
    :return:
    """
    # write mhd/NIFTI image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)


# 自定义error, 方便报错的时候显示
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ShapeError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class VisError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


# show2D(mat2gray(gaussian_filter([81,81], 30)))


def gaussian_filter(size, sigma = 10):
    """
    制作高斯分布的矩阵
    (注意考虑边界的情况），单个单个的元素处理会不会有些慢呢？
    分为1D和2D的情况，这里只允许size是奇数
    :param size: list contains 2 or 1 integer, or only one integer, 1 or [2,3] or [1]  2D filter,
    :param sigma: 越大，越偏平，中间层越不突出
    :return:

    example:
        show2D(mat2gray(gaussian_filter([81,81], 30)))
        show1D(mat2gray(gaussian_filter(81,30)))

    """

    # 制作高斯核存放的矩阵
    if isinstance(size, list): # 判断是list还是integer
        dim_num = len(size)
        if dim_num == 2:
            # 检查是否为奇数
            if [i for i in size if i % 2 == 0]:
                raise ValueError(r'The element in list"size" should be odd')
            gaussian = np.zeros([size[0], size[1]])
        else:
            # 检查是否为奇数
            if size[0] % 2 == 0:
                raise ValueError(r'The element in list"size" should be odd')
            gaussian = np.zeros([size[0], size[0]])
    elif isinstance(size, int):
        dim_num = 1
        if size%2 ==0:
            raise ValueError(r'The "size" should be odd')
        gaussian = np.zeros([size, size])
    else:
        raise ValueError(r'The "size" should be list or integer')

    # perform
    sigma1 = sigma2 = sigma
    gau_sum = 0
    center = [int(gaussian.shape[0]*0.5),int(gaussian.shape[1]*0.5)]
    gaussian = np.zeros([size, size])
    for i in range(gaussian.shape[0]):
        for j in range(gaussian.shape[1]):
            gaussian[i, j] = math.exp((-1 / (2 * sigma1 * sigma2)) * (np.square(i - center[0])
                                       + np.square(j - center[1])))\
                                      / (2 * math.pi * sigma1 * sigma2)
            gau_sum = gau_sum + gaussian[i, j]

    # show2D(mat2gray(gaussian))

    # 根据2D or 1D，返回vector or matrix
    if dim_num == 2:
        return gaussian.astype(np.float32)
    elif dim_num == 1:
        return (gaussian[center[0]]).astype(np.float32)


# plt.plot(gaussian_filter(7, 1.6))
# plt.show()


def checkoutIndex(array3D, index):
    """

    :param array3D:
    :param index: [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    :return:
    """
    d0_lt, d1_lt, d2_lt = list(array3D.shape)  # d0_lt, 即dim 0 limitation
    # 判断是否超过上界
    if (index[0] > d0_lt or
        index[1] > d0_lt or
        index[2] > d1_lt or
        index[3] > d1_lt or
        index[4] > d2_lt or
        index[5] > d2_lt):
        return False

    # 判断是否超过下界
    negative_num = list(filter(lambda x:x<0, index))
    if negative_num: # 如果存放负数的数组不为空，则有越界的
        return False

    # 判断是否顺序颠倒或索引相同，比如[23,13]or[23,23]是不可的
    if (index[0] >= index[1] or
        index[2] >= index[3] or
        index[4] >= index[5] ):
        return False

    return True

def bbox_scale(bbox, trans_rate):
    """
    因为原点是【0，0】，所以坐标直接缩放即可
    :param bbox: 坐标 [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    :param trans_rate: (dim0r,dim1r,dim2r)
    :return: 注意，这里返回的坐标不是整数，只有显示的时候才是整数，网络预测出来的坐标以及计算loss的坐标都不是整数
    """
    trans_rate = list(trans_rate)
    trans_rate = [trans_rate[0], trans_rate[0], trans_rate[1],trans_rate[1],trans_rate[2],trans_rate[2]]
    trans_rate = np.array(trans_rate)
    return list(np.array(trans_rate)* np.array(bbox))


def make_bbox_square(bbox):
    """
    把bbox变成正方体的bbox，但是不改变原bbox中心
    :param bbox: [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    :return: 注意，返回的bbox不是整数！bbox就是bbox，可以不是整数！
    """
    max_lenth = np.max([bbox[1]-bbox[0],bbox[3]-bbox[2],bbox[5]-bbox[4]])
    center = [(bbox[1]+bbox[0])*0.5,(bbox[3]+bbox[2])*0.5,(bbox[5]+bbox[4])*0.5]
    new_bbox = [(center[0] - max_lenth * 0.5), (center[0]+max_lenth*0.5),
                (center[1] - max_lenth * 0.5), (center[1]+max_lenth*0.5),
                (center[2] - max_lenth * 0.5), (center[2]+max_lenth*0.5)]
    return new_bbox


def slice_neibor_add_one_dim(scan,  axis, add_num, add_weights, g_sigma):
    """

    :param scan: 3D adarray， axis必须是以下顺序（'coronal','sagittal','axial', 或'x','y','z', 或0, 1, 2）
    :param axis: 'coronal','sagittal','axial', 或'x','y','z', 或0, 1, 2
    :param add_num: 奇数
    :param add_weights: ‘Gaussian’，‘Mean’， ‘DeGaussian’（即1-maxminscale（Gaussian））
    :return:
    """

    # 首先得到1D的权重vector
    if add_weights == 'Mean':
        weights = np.ones(add_num, dtype=np.float32)
    elif add_weights == 'Gaussian':
        weights = gaussian_filter(add_num, g_sigma)
    elif add_weights == 'DeGaussian':
        weights = gaussian_filter(add_num, g_sigma)
        weights = np.max(weights)- weights + np.min(weights)
    else:
        raise ValueError

    # 将要叠加的轴挪到第一个位置
    if axis == 'coronal' or axis == 'x' or axis == 0:
        pass  # 已经在第一维，没什么好做的
    elif axis == 'sagittal' or axis == 'y' or axis == 1:
        scan = np.transpose(scan, (1, 2, 0))
    elif axis == 'axial' or axis == 'z' or axis == 2:
        scan = np.transpose(scan, (2, 1, 0))
    else:
        raise ValueError


    # 获取第一维度坐标的上界
    dim1max = scan.shape[0] - 1 # 减一是因为我们需要的是坐标的上界

    # 新建一个诡异的矩阵，用来存放最终的结果
    tmp_array = np.zeros(scan.shape, dtype=scan.dtype)

    # 将相邻层叠加到此层（其实按理来说，用kernel去卷积会更快，这里选择比较蠢的实现方式）
    for i in range(dim1max+1):  # 这里+1是因为，dim1max是索引的上界，但是range应该加len， 而len = dim1max+1
        # if i == 287:
        #     break
        # 首先取出相邻的add_num的层（add_num=5，则向左取2层，向右取2层）
        min_index, max_index= [i- add_num//2, i+ add_num//2]
        min_index, max_index = [np.max([0, min_index]), np.min([dim1max, max_index])] # 防止越界
        tmp_slices = scan[min_index:max_index+1, :, :]

        # 修正权重（因为tmp_slices在最上层，取不到更上层，所以这时候需要舍弃一些权重）
        if min_index == 0: # 图像左侧被舍弃，则权重从右开始取权重
            weights_ = ((weights[::-1])[min_index:max_index+1])[::-1]  # 逆序、截取、逆序
        elif max_index == dim1max and (add_num - tmp_slices.shape[0]) != 0:  #
            weights_ = weights[:-(add_num - tmp_slices.shape[0])]
        else:
            weights_ = weights

        weights_ = np.expand_dims(weights_, axis=1)  # expand for broadcast
        weights_ = np.expand_dims(weights_, axis=1)

        # 所有层加权平均
        tmp_slices_ = np.sum(weights_*tmp_slices, axis=0)/ np.sum(weights_)

        # 赋值
        tmp_array[i,:,:] = tmp_slices_

    # 将轴的顺序还原
    if axis == 'coronal' or axis == 'x' or axis == 0:
        pass  # 已经在第一维，没什么好做的
    elif axis == 'sagittal' or axis == 'y' or axis == 1:
        tmp_array = np.transpose(tmp_array, (2, 0, 1))  # 从（1，2，0） 还原到（0，1，2）
    elif axis == 'axial' or axis == 'z' or axis == 2:
        tmp_array = np.transpose(tmp_array, (2, 1, 0))  # 从（2，1，0） 还原到（0，1，2）

    return tmp_array


class wama():
    """
    以病人为单位的class
    1) 包含图像与标注
    2）不要有复杂操作或扩增，这些应该另外写代码，否则会占用大量内存？
    3) 包含简单的预处理，如调整窗宽窗位，resampleling

    param：
    patch_mode：
        滑动窗slideWin模式：连续多层，可以加expansion rate（类似空洞卷积），参数还有滑动步长
        风车windmill模式：将某平面的中心点垂线作为旋转轴，旋转采样，采样结果可能为3D和2D，若为2D，可能需要优化一下，比如相邻层和本层叠加

    """
    def __init__(self):
        """
        只支持单肿瘤
        """
        # 可能会用到的一些信息
        self.id = None
        # 存储图像的信息
        self.scan = {}  # 字典形式储存数据，如image['CT']=[1,2,3]， 不同模态的图像必须要是配准的！暂时不支持没配准的
        self.spacing = {}  # 字典形式存储数据的,tuple
        self.origin = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息 todo
        self.transfmat = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息
        self.axesOrder = {}  # 字典形式存储数据的, ??，注意，mask不需要这个信息

        self.resample_spacing = {}  # tuple, 一旦存在，则表示图像已经经过了resample

        # 储存mask，只需储存图像即可
        self.sementic_mask = {}  # 同上，且要求两者大小匹配，暂时只支持一个病人一个肿瘤（否则在制作bbox的时候会有问题）
        self.bbox = {}  # 将mask取最小外接方阵，或自己手动设置


        # 分patch的操作，在外面进行，反正只要最后可以还原就行了
        # 储存分patch的信息（要考虑分patch出现2D和3D的情况）,分patch的时候记得演示分patch的过程
        self.is_patched = False # 是否进行了分patch的操作  （每次添加了新的数据、模态、mask，都需要将这个设置为False，之后重新分patch）
        self.patch_mode = None  # 分patch的模式，暂时包括：滑动窗slideWin，风车windmill，不同模式对应不同的参数
        self.patch_config = {}  # 分patch的详细参数
        self.patch_num = None   # patch的数量
        self.patch = []  # 直接储存patch到list


    """从NIFTI加载数据系列"""
    def appendImageFromNifti(self, img_type, img_path, printflag = False):
        """
        添加影像
        :param img_type:
        :param img_path:
        :param printflag: 是否打印影像信息
        :return:
        """
        # 首先判断是否已有该模态（img_type）的数据
        if img_type in self.scan.keys():
            warnings.warn(r'alreay has type "' + img_type + r'", now replace it')
        # 读取数据
        scan, spacing, origin, transfmat, axesOrder = readIMG(img_path)
        # 存储到对象
        self.scan[img_type] = scan
        self.spacing[img_type] = spacing
        self.origin[img_type] = origin
        self.transfmat[img_type] = transfmat
        self.axesOrder[img_type] = axesOrder
        if printflag:
            print('img_type:',img_type)
            print('img_shape:',self.scan[img_type].shape)
            print('spacing',self.spacing[img_type])
            print('axesOrder',self.axesOrder[img_type])

    def appendSementicMaskFromNifti(self, img_type, mask_path):
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 读取mask
        mask, _, _, _, _ = readIMG(mask_path)
        # 检查形状是否与对应img_type的scan一致
        if mask.shape != self.scan[img_type].shape:
            raise ShapeError(r'shape Shape mismatch error: scan "' + img_type + \
                             r'" shape is'+ str(self.scan[img_type].shape)+ \
                             r', but mask shape is '+ str(mask.shape))

        # 将mask存入对象
        self.sementic_mask[img_type] = mask

    def appendImageAndSementicMaskFromNifti(self, img_type, img_path, mask_path, printflag = False):
        self.appendImageFromNifti(img_type, img_path, printflag)
        self.appendSementicMaskFromNifti(img_type, mask_path)


    """读取数据"""
    # 获取整个图像
    def getImage(self, img_type):
        """

        :param img_type:
        :return:  ndarray of whole_size img
        """
        return deepcopy(self.scan[img_type])

    # 获取整个mask
    def getMask(self, img_type):
        """

        :param img_type:
        :return: ndarray of whole_size mask
        """
        return deepcopy(self.sementic_mask[img_type])

    # 获取bbox内的图像
    def getImagefromBbox(self, img_type, ex_voxels=0, ex_mms=None, ex_mode='bbox', aim_shape=None):
        """
        先用mask和原图点乘，之后外扩一定体素的bbox取出来（注意，各个维度外扩的尺寸是固定的，暂时）,
        :param img_type:
        :param ex_voxels: 一个值！不要乱搞乱赋值，ex_voxels = 20 这样子
        :param ex_mms: 指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到原图
        mask_roi_img = self.scan[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=0)

        return roi_img

    # 获取mask内的图像
    def getImagefromMask(self, img_type, ex_voxels=0, ex_mms=None, ex_mode ='bbox', aim_shape = None):
        """
        先用mask和原图点乘，之后外扩一定体素的bbox取出来（注意，各个维度外扩的尺寸是固定的，暂时）,
        :param img_type:
        :param ex_voxels: 一个值！不要乱搞乱赋值，ex_voxels = 20 这样子
        :param ex_mms: 指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 用mask和原图点乘
        mask_roi_img = self.scan[img_type] * self.sementic_mask[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=0)

        return roi_img

    # 获取bbox内的mask
    def getMaskfromBbox(self, img_type, ex_voxels=0, ex_mms=None, ex_mode='bbox', aim_shape=None):
        """
        先用mask和原图点乘，之后外扩一定体素的bbox取出来（注意，各个维度外扩的尺寸是固定的，暂时）,
        :param img_type:
        :param ex_voxels: 一个值！不要乱搞乱赋值，ex_voxels = 20 这样子
        :param ex_mms: 指定外扩的尺寸(优先级最高，一旦有此参数，忽略ex_voxels）
        :param ex_mode:'bbox' or 'square', bbox则直接在bbox上外扩，square则先变成正方体，再外扩(注意，由于外扩后可能index越界，所以不一定是正方体）
        :param aim_shape: e.p. [256, 256, 256]
        :return: array of Mask_ROI
        """

        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到mask
        mask_roi_img = self.sementic_mask[img_type]

        # 得到bbox
        bbox = self.bbox[img_type]

        # 按照ex_mode，选择是否把bbox变成立方体
        if ex_mode == 'square':
            bbox = make_bbox_square(bbox)
            print('make_bbox_square')

        # 计算需要各个轴外扩体素
        ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
        if ex_mms is not None:  # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
            if self.is_resample(img_type):
                ex_voxels = [ex_mms / i for i in list(self.resample_spacing[img_type])]
            else:
                ex_voxels = [ex_mms / i for i in list(self.spacing[img_type])]

        # 外扩体素（注意，滑动的轴不外扩）
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]

        # bbox取整
        bbox = [int(i) for i in bbox]

        # 检查是否越界
        bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
        bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], mask_roi_img.shape[0]]),
                                     np.min([bbox[3], mask_roi_img.shape[1]]),
                                     np.min([bbox[5], mask_roi_img.shape[2]])]

        # 将图像抠出
        roi_img = mask_roi_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # 如果有aim_shape,则返回resize后的
        if aim_shape is not None:
            roi_img = resize3D(roi_img, aim_shape, order=0)

        return roi_img


    #
    def getBbox(self, img_type):
        # 首先检查是不是有bbox(有bbox必定有mask和img）
        if img_type not in self.bbox.keys():
            self.make_bbox_from_mask(img_type)

        # 得到bbox
        bbox = self.bbox[img_type]

        return bbox

    # """从Array加载数据系列"""
    # def appendImageFromArray(self, img_type ,img_array, voxel_size):
    #
    #
    # def appendSementicMaskFromArray(self, img_type, mask_path):
    #     self.shape_check()
    #
    # def appendImageAndSementicMaskFromArray(self, img_type, img_path, mask_path):


    """基于mayavi的可视化"""
    def show_scan(self, img_type, show_type = 'volume'):
        """

        :param img_type:
        :param show_type: volume or slice
        :return:
        """
        # 检查是否存在对应模态
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查是否安装了mayavi
        if mayavi_exist_flag:
            if show_type == 'volume':
                show3D(self.scan[img_type])
            if show_type == 'slice':
                show3Dslice(self.scan[img_type])
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_mask(self, img_type, show_type = 'volume'):
        """

        :param img_type:
        :param show_type: volume or slice
        :return:
        """
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        if mayavi_exist_flag:
            if show_type == 'volume':
                show3D(self.sementic_mask[img_type])
            if show_type == 'slice':
                show3Dslice(self.sementic_mask[img_type])
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_maskAndImage(self, img_type, show_type = 'volume'):
        """
        拼接在一起显示
        :param img_type:
        :param show_type:
        :return:
        """
        # 只检查mask即可，因为有mask必有image
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        if mayavi_exist_flag:
            # 读取mask和image，并拼接
            mask = self.sementic_mask[img_type]
            image = self.scan[img_type]
            image_mask = np.concatenate([image,mask],axis=2)


            if show_type == 'volume':
                show3D(image_mask)
            if show_type == 'slice':
                show3Dslice(image_mask)
            else:
                raise VisError('only volume and slice mode is allowed')
        else:
            warnings.warn('no mayavi exsiting')

    def show_bbox(self, img_type, show_type='volume'):
        """
        显示bbox，
        :param img_type:
        :param show_type:
        :return:
        """
        raise NotImplementedError

    def show_bbox_with_img(self, img_type, show_type='volume'):
        """
        显示bbox内的图像
        :param img_type:
        :param show_type:
        :return:
        """
        raise NotImplementedError

    """ annotation操作 """
    def make_bbox_from_mask(self, img_type, big_connnection = False):
        """
        目前只支持单肿瘤
        :param img_type:
        big_connnection: 是否基于最大连通域，如果是粗标注，则设为False
        :return:
        """

        # 检查对应的img_type是否有mask
        if img_type not in self.sementic_mask.keys():
            warnings.warn(r'you need to load "' + img_type + r'" mask first')
            return

        # 提取mask
        mask = self.sementic_mask[img_type]

        # 若只取最大连通域，则执行取最大连通域操作
        if big_connnection:
            mask = connected_domain_3D(mask)

        # 计算得到bbox，形式为[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
        indexx = np.where(mask> 0.)
        dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                                np.min(indexx[1]), np.max(indexx[1]),
                                                                np.min(indexx[2]), np.max(indexx[2])]
        self.bbox[img_type] = [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]

    def add_box(self, img_type, bbox):
        """
        ！！ 需要在resample操作前进行，一旦经过了resample，就不可以添加Bbox了（我是不相信你会自己去算😊）
        :param bbox: 要求按照此axis顺序给出  coronal,sagittal,axial （或x,y,z）
                    example ：[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
        """
        # 检查是否有对应img_type的图像
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查坐标是否超出范围
        if checkoutIndex(self.scan[img_type], bbox):
            raise IndexError('Bbox index out of rang')

        # 加入坐标
        self.bbox_mask[img_type] = bbox
        # 利用坐标生成mask， 方便resample的操作
        # 储存mask

    def get_bbox_shape(self, img_type):
        """返回肿瘤的大小: 即lenth_dim0到2， list
            注意，返回voxel number， 同时返回true size（mm^3），（cm^3）
        """
        # 先看看有妹有bbox，有就直接搞出来
        if img_type in self.bbox.keys():
            print('get from bbox')
            bbox = self.bbox[img_type]
            return [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]

        # 妹有就看看有妹有mask，有就直接调用，注意连通域函数
        if img_type in self.sementic_mask.keys():
            # 得到bbox
            print('making bbox')
            self.make_bbox_from_mask(img_type)
            # 返回shape
            print('get from bbox')
            bbox = self.bbox[img_type]
            return [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]

        # 啥都没有，就算了
        warnings.warn(r'you need to load "' + img_type + r'" mask or bbox first')
        return

    def get_scan_shape(self, img_type):
        return self.scan[img_type].shape




    """prepocessing"""
    def resample(self, img_type, aim_spacing): # TODO
        """

        :param img_type:
        :param aim_space: tuple with 3 elements (dim0, dim1, dim2), or 1 interger
        :return:
        """
        # 原图、mask、bbox都需要！！！，bbox可以先转为矩阵，然后resize后重新获得坐标

        # 检查是否有对应的image
        if img_type not in self.scan.keys():
            warnings.warn(r'you need to load "' + img_type + r'" scan or image first')
            return

        # 检查aim_spacing todo

        # 首先计算出各个轴的scale rate （这里要确保scan和spacing的dim是匹配的）
        # 这里需要注意！：如果已经经过了resample，那么需要将最后一此resample的spacing作为当前的spacing todo
        if self.is_resample(img_type):
            or_spacing = self.resample_spacing[img_type]
        else:
            or_spacing = self.spacing[img_type]
        trans_rate = tuple(np.array(or_spacing)/np.array(aim_spacing))

        # resample， 并记录aim_spacing, 以表示图像是经过resample的
        # 记录aim_spacing
        self.resample_spacing[img_type] = aim_spacing
        # 先对原图操作
        self.scan[img_type] = zoom(self.scan[img_type], trans_rate,order=2) # 用双三次插值？
        # 再对mask操作
        if img_type in self.scan.keys():
            self.sementic_mask[img_type] = zoom(self.sementic_mask[img_type], trans_rate, order=0)  # 最近邻插值？（检查下是不是还是二值图接可）
        # 再对BBox操作（转化为mask，之后resize，之后取bbox）
        if img_type in self.bbox.keys():
            self.bbox[img_type] = bbox_scale(self.bbox(img_type),trans_rate) # todo 需要检查一下

    def is_resample(self, img_type):
        """
        判断图像是否经过resample, 若已经经过重采样，则返回True
        :param img_type:
        :return:
        """
        if  img_type in self.resample_spacing.keys():
            return True
        else:
            return False

    def adjst_Window(self,img_type, WW, WL):
        """
        调整窗宽窗位
        :param img_type: 图像种类
        :param WW: 窗宽
        :param WL: 窗位
        :return:
        """
        self.scan[img_type] = adjustWindow(self.scan[img_type], WW, WL)

    def slice_neibor_add(self, img_type, axis = ['axial'], add_num = [5], add_weights = 'Gaussian', g_sigma = 3):
        """
        任何时候操作都可以，只能对scan操作
        slice neighbor add, 相邻层累加策略，类似 mitk 里面的那个多层叠加显示的东西，等价于平滑

        指定1个axis， ok ，那么只在这一个axis操作
        如果2个，则各自在1个axis操作，之后2个操作后的矩阵取平均
        3个也同理

        ！！ 直观上讲，最好resample到voxel为正方形再搞，不过实际上是无所谓
        :param img_type:
        :param axis: list, can be ['coronal','sagittal','axial'], ['x','y','z'], [0, 1, 2]
        :param add_num: list, 维度要和axis匹配，且list中的element必须是正奇数
        :param add_weights: ‘Gaussian’，‘Mean’， ‘DeGaussian’（即1-maxminscale（Gaussian））
        :param g_sigma: ‘Gaussian’或‘DeGaussian’模式下的方差，越大权重越接近于mean
        :return:
        """

        # 用来存放各个变换后的图像
        tmp_scan_list = []

        # 逐个完成变换
        for index, _axis in enumerate(axis):
            tmp_scan_list.append(slice_neibor_add_one_dim(self.scan[img_type], _axis, add_num[index], add_weights, g_sigma))
            # tmp_scan_list.append(slice_neibor_add_one_dim(mask, _axis, add_num[index], add_weights, g_sigma))

        # 将变换后的所有图像取平均， 重新赋予
        if len(tmp_scan_list)== 1:
            return tmp_scan_list[0]
        elif len(tmp_scan_list)==2:
            return (tmp_scan_list[0]+tmp_scan_list[1])/2
        elif len(tmp_scan_list)==3:
            return (tmp_scan_list[0]+tmp_scan_list[1]+tmp_scan_list[2])/3

    def _normalization(self, img_type):
        self.scan[img_type] = standardization(self.scan[img_type])

    def _minmaxscale(self, img_type):
        self.scan[img_type] = mat2gray(self.scan[img_type])


    """postprocessing"""
    #这个操作可以挪到外面，因为最后还是要分开保存
    def makePatch(self, mode, **kwargs):
        """
        逻辑：首先框取ROI，之后在ROI内进行操作
        :param mode: 'slideWin' or 'windmill'
        :param kwargs:
                aim_shape: 基于GT or BBox坐标 截取固定大小的ROI进行分patch，最后会有3种情况，即肿瘤mask大于、等于、小于这个ROI， 若不指定，则直接为GT 或 BBox大小
                ex_voxels：在aim_shape基础上外扩一定像素

                不同mode对应不同参数: 滑动窗'slideWin'，风车'windmill'
                @ slideWin对应的参数：(一般使用这个操作，可以尝试辅助slice_nb_add这个操作)
                    size: list, e.p. [80, 80],  or [80, 80, 16]， 即滑动框的shape
                    axis: list, 滑动的轴，['coronal', 'sagittal', 'axial'], or ['sagittal', 'axial'] or ['x','y','z'] or ['dim0', 'dim1', dim2'] or [0, 1, 2]
                                这里需要注意，有如下对应关系：
                                当滑动轴为1个时，window的size必须有两个维度是与原图尺寸相同的，即原图[80,80,80],size其中两维至少为80，这样才能保证在一个axis滑动
                                同理，滑动轴为2个时，必须size中一维为最大值
                                滑动轴为3个时，size任意一维度都不要大于最大值
                    stride: list， 指定几个滑动轴，就有几个步长

                    返回一个list， 所有patch的坐标列表，坐标形式和scan的axis[x,y,z]对应
                    还会返回patch的数量
                    以上都会储存在类里

                @ windmill 对应的参数 (一般使用这个操作，不需要辅助slice_nb_add这个操作)
                    axis：滑动的轴，只可指定一个，['coronal', 'sagittal', 'axial'], or ['sagittal', 'axial'] or ['x','y','z'] or ['dim0', 'dim1', dim2'] or [0, 1, 2]
                    add_num: 相邻层累加到当前层的厚度，指定为0则不累积，指定为1则当前层n会加上相邻n-1和n+1层的信息
                    add_weights: list，累加时候相邻层的权重，不设置则直接平均，例n为3时，可设置为[3,2,1,1],对应[n,n+1,n+2,n+3]的权重
                    patch_slice: 即patch的厚度，默认为1

        """
        raise NotImplementedError


"""patch的操作"""

# 滑动窗还原patch的操作极其简单，只需要赋patch予到原始空间位置即可
# 分patch的操作包括：基于bbox, 外扩一定像素，resize到目标大小，分patch，储存patches
def _slide_window_one_axis(array3D,
                           spacing,
                           origin,
                           transfmat,
                           axesOrder,
                           bbox,
                           axis,
                           slices = 1,
                           stride = 1,
                           expand_r = 1,
                           mask = None,
                           ex_mode = 'bbox',
                           ex_voxels = 0,
                           ex_mms = None,
                           resample_spacing=None,
                           aim_shape = 256,
                           ):
    """

    :param array3D: 3D 图像
    :param spacing: 图像的原始spacing，对应[coronal,sagittal,axial]
    :param origin: 原始图像的realworld原点（sitk读图时会返回）
    :param transfmat: 原始图像的坐标系基向量（所谓的空间方向向量）
    :param axesOrder: 3D array的坐标轴的顺序
    :param resample_spacing: 重采样后的spacing, list

    :param bbox: 肿瘤orROI的bbox（注意是最小外接立方体，但是可能不是正方体，且与坐标系轴平行）
    :param axis: 分patch沿着的轴, 可以指定为['coronal','sagittal','axial']， or['x','y','z'], or [0,1,2] or ['dim0', 'dim1', dim2']
    :param slices: 层数，即patch的层厚
    :param stride: 每次滑动的步长
    :param expand_r: slice间的膨胀因子，1则不膨胀,2 则每隔1层取一层，依此类推
    :param mask: 如果有分割的图，可一起输入分patch
    :param ex_mode: 外扩的模式，一个是在最小外界矩阵直接外扩'bbox'，一个是先变成“正方体”再外扩'square'
    :param ex_voxels: 外扩的像素数（一个整数）
    :param ex_mms: 外扩的尺寸,单位mm（优先级比较高，可以不指定ex_voxels而是指定这个， 当ex_voxels和ex_mms同时存在时，只看ex_mms）
    :param aim_shape:一个整数（默认，强制，输出patch的那个面，是正方形）
    :return:
    """


    # 按照ex_mode，选择是否把bbox变成立方体
    if ex_mode ==  'square':
        bbox = make_bbox_square(bbox)

    # 计算需要各个轴外扩体素
    ex_voxels = [ex_voxels, ex_voxels, ex_voxels]
    if ex_mms is not None: # 如果有ex_mms，则由ex_mms生成list格式的ex_voxels
        if resample_spacing is not None:
            ex_voxels = [ex_mms/i for i in list(resample_spacing)]
        else:
            ex_voxels = [ex_mms / i for i in list(spacing)]

    # 外扩体素（注意!!!!，滑动的轴不外扩）
    if axis == 'coronal'  or axis == 'x' or axis == 'dim0' or axis == 0:
        bbox = [bbox[0], bbox[1],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]
    elif axis == 'sagittal' or axis == 'y' or axis == 'dim1' or axis == 1:
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2], bbox[3],
                bbox[4] - ex_voxels[2], bbox[5] + ex_voxels[2]]
    elif axis == 'axial'    or axis == 'z' or axis == 'dim2' or axis == 2:
        bbox = [bbox[0] - ex_voxels[0], bbox[1] + ex_voxels[0],
                bbox[2] - ex_voxels[1], bbox[3] + ex_voxels[1],
                bbox[4], bbox[5]]

    # bbox取整
    bbox = [int(i) for i in bbox]

    # 检查是否越界
    bbox[0], bbox[2], bbox[4] = [np.max([bbox[0], 0]), np.max([bbox[2], 0]), np.max([bbox[4], 0])]
    bbox[1], bbox[3], bbox[5] = [np.min([bbox[1], array3D.shape[0]]),
                                 np.min([bbox[3], array3D.shape[1]]),
                                 np.min([bbox[5], array3D.shape[2]])]



    # 抠出肿瘤，有mask则一起操作
    _scan = array3D[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    if mask is not None:
        _mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]


    # resize到目标shape，也就是aim_shape,
    # （注意，指定x轴，则y、z轴resize到aim_shape，但由于yz面可能不是正方形，所以暂时取yz面边长"平均数"计算x轴缩放比例）
    if aim_shape is not None:
        if axis == 'coronal'  or axis == 'x' or axis == 'dim0' or axis == 0:
            mean_lenth = (_scan.shape[1]+_scan.shape[2])/2  # 取均值
            _scan = zoom(_scan, (aim_shape/ mean_lenth, aim_shape/_scan.shape[1], aim_shape/_scan.shape[2]), order=3) # cubic插值
            if mask is not None:
                _mask = zoom(_mask, (aim_shape / mean_lenth, aim_shape / _scan.shape[1], aim_shape / _scan.shape[2]), order=0)  # nearest插值
        elif axis == 'sagittal' or axis == 'y' or axis == 'dim1' or axis == 1:
            mean_lenth = (_scan.shape[0]+_scan.shape[2])/2  # 取均值
            _scan = zoom(_scan, (aim_shape/_scan.shape[0] , aim_shape/mean_lenth, aim_shape/_scan.shape[2]), order=3) # cubic插值
            if mask is not None:
                _mask = zoom(_mask, (aim_shape/_scan.shape[0] , aim_shape/mean_lenth, aim_shape/_scan.shape[2]), order=0)  # nearest插值
        elif axis == 'axial'    or axis == 'z' or axis == 'dim2' or axis == 2:
            mean_lenth = (_scan.shape[0]+_scan.shape[1])/2  # 取均值
            _scan = zoom(_scan, (aim_shape/_scan.shape[0] , aim_shape/_scan.shape[1], aim_shape/mean_lenth), order=3) # cubic插值
            if mask is not None:
                _mask = zoom(_mask, (aim_shape/_scan.shape[0] , aim_shape/_scan.shape[1], aim_shape/mean_lenth), order=0)  # nearest插值


    # 开始分patch，并且保存每个patch所在的index，stride，以备复原 todo
    patches = []
    # 首先将目标轴移动到第一个, 进行分patch，并保存信息
    roi_scan_shape = _scan.shape  # 未经过axis调整，axis order和原图一致时的roi的shape
    # 将要叠加的轴挪到第一个位置
    if axis == 'coronal' or axis == 'x' or axis == 0:
        pass  # 已经在第一维，没什么好做的
    elif axis == 'sagittal' or axis == 'y' or axis == 1:
        _scan = np.transpose(_scan, (1, 2, 0))
        if mask is not None:
            _mask = np.transpose(_mask, (1, 2, 0))
    elif axis == 'axial' or axis == 'z' or axis == 2:
        _scan = np.transpose(_scan, (2, 1, 0))
        if mask is not None:
            _mask = np.transpose(_mask, (2, 1, 0))
    else:
        raise ValueError


    # 分patch，patch的data，以及其他shape，index信息保存在patch的类里面，patch.data以及patch.info(结构为字典），之后用pickle打包patch存储即可
    # 记得将每一个patch的轴还原

    # 先沿着分patch的轴，滑动，滑动的stride（也叫steps）就是参数的stride
    for i in range(0, _scan.shape[0], stride):
        # 现在每个i其实就是一个起点，根据这个起点，采样slices个层，间隔位expand_r
        # 首先采样：这里为什么使用i到i+(slices*expand_r)这个范围，自己好好琢磨下即可（应该是没问题的）
        _tmp_patch_array = _scan[i:i+(slices*expand_r):expand_r, :, :]  # 放心，就算只取一层，也会是（1，w，h）的shape
        if mask is not None:
            _tmp_mask_array = _mask[i:i+(slices*expand_r):expand_r, :, :]


        # 因为ndarray采样越界也不会报错，so需要进一步
        # 判断采样出来的array层数是否等于slices，如果小于则证明已经“采到头了”，则break出循环
        if _tmp_patch_array.shape[0] < slices:
            break
        else:  #如果patch尺寸合格，则储存
            # 将轴的顺序还原patch
            if True:
                if axis == 'coronal' or axis == 'x' or axis == 0:
                    pass  # 已经在第一维，没什么好做的
                elif axis == 'sagittal' or axis == 'y' or axis == 1:
                    _tmp_patch_array = np.transpose(_tmp_patch_array, (2, 0, 1))  # 从（1，2，0） 还原到（0，1，2）
                    if mask is not None:
                        _tmp_mask_array = np.transpose(_tmp_mask_array, (2, 0, 1))
                elif axis == 'axial' or axis == 'z' or axis == 2:
                    _tmp_patch_array = np.transpose(_tmp_patch_array, (2, 1, 0))  # 从（2，1，0） 还原到（0，1，2）
                    if mask is not None:
                        _tmp_mask_array = np.transpose(_tmp_mask_array, (2, 1, 0))

            # 储存数据到对象
            if True:
                _tmp_patch = patch_tmp()  # 先建个对象储存patch的数据
                _tmp_patch.data = _tmp_patch_array  # 储存patch图像
                if mask is not None:
                    _tmp_patch.mask = _tmp_mask_array

            # 接下来尽可能的保存info，已备还原patch
            if True:
                _tmp_patch.info['patch_mode'] = r'_slide_window_one_axis'  # 记录分patch的模式
                # 记录数据：首先要还原到分patch之前的_scan需要的信息有以下
                _tmp_patch.info['axis'] = axis
                _tmp_patch.info['slices'] = slices
                _tmp_patch.info['expand_r'] = expand_r
                _tmp_patch.info['index_begin'] = i
                _tmp_patch.info['_scan.shape'] = roi_scan_shape  # 需要是未经过axis调整（即目标axis提前到第一轴）的shape
                # 记录数据：之后需要从_scan还原到原图，需要
                _tmp_patch.info['_scan_bbox'] = bbox  # aim_shape缩放之前的bbox（bbox可以计算出shape）
                _tmp_patch.info['array3D.shape'] = array3D.shape  # 最原始大图的shape
                _tmp_patch.info['array3D.spacing'] = spacing  # 最原始大图的spacing
                _tmp_patch.info['array3D.resample_spacing'] = resample_spacing  # 最原始大图的resample_spacing(如果不是None，则以此为准，此spacing的优先级最高）
                _tmp_patch.info['array3D.origin'] = origin  # 最原始大图的origin
                _tmp_patch.info['array3D.transfmat'] = transfmat  # 最原始大图的origin
                _tmp_patch.info['array3D.axesOrder'] = axesOrder  # 最原始大图的axesOrder，也是_scan、最终patch中data的axesOrder

            # 将对象存入list
            patches.append(_tmp_patch)

        # 注意，这里我们不从后往前取一个patch，主要原因是我懒得写代码了，（但是这可能会对分割任务有影响）
        # （因为分割金标准不能随便丢，so 分割任务的stride建议为1， 或 axis_len - slices 能被 stride整除）
        # so，直接返回patches的liest



    # 把_scan的axis顺序也调整回去
    if True:
        if axis == 'coronal' or axis == 'x' or axis == 0:
            pass  # 已经在第一维，没什么好做的
        elif axis == 'sagittal' or axis == 'y' or axis == 1:
            _scan = np.transpose(_scan, (2, 0, 1))  # 从（1，2，0） 还原到（0，1，2）
        elif axis == 'axial' or axis == 'z' or axis == 2:
            _scan = np.transpose(_scan, (2, 1, 0))  # 从（2，1，0） 还原到（0，1，2）


    return patches

def _slide_window_one_axis_reconstruct(patches):
    """
    暂时只做到还原image，ok？（如果想还原mask,再说）
    ps:
    重建的时候，需要注意，如果patch是有重叠的，那么重复赋值之后，需要把赋值次数为n的体素，除以n以获得均值
    我们可以额外建立一个数组（值全部为1的矩阵），作为储存各个体素被赋值次数的矩阵，最后再除以这个矩阵即可

    :param patches_list: patch对象组成的list
    :return:
    """

    # 构建个容器
    img = np.zeros(patches[0].info['_scan.shape'],dtype=np.float32)
    weight_img = np.zeros(patches[0].info['_scan.shape'],dtype=np.float32)

    # 获取axis
    axis = patches[0].info['axis']

    # 逐个patch放回咯
    for patch in patches:
        i = patch.info['index_begin']
        slices = patch.info['slices']
        expand_r = patch.info['expand_r']

        if axis == 'coronal' or axis == 'x' or axis == 0:
            img[i:i + (slices * expand_r):expand_r, :, :] += patch.data
            weight_img[i:i + (slices * expand_r):expand_r, :, :] += 1.
        elif axis == 'sagittal' or axis == 'y' or axis == 1:
            img[:,i:i + (slices * expand_r):expand_r, :] += patch.data
            weight_img[:,i:i + (slices * expand_r):expand_r, :] += 1.
        elif axis == 'axial' or axis == 'z' or axis == 2:
            img[:,:,i:i + (slices * expand_r):expand_r] += patch.data
            weight_img[:,:,i:i + (slices * expand_r):expand_r] += 1.


    # weight_img初始化为0，记录次数，但是可能会有0的存在，所以要修正0为1，因为0次赋值和1次赋值的weight都应该是1
    weight_img[weight_img<0.5] = 1.


    img_final = img / weight_img  # todo  感觉哪里不太对的鸭子



    # 暂时只做到重建_scan，返回
    return img

show3Dslice(np.concatenate([_scan,img],axis=1))
show3Dslice(np.concatenate([_scan,img_final],axis=1))
show3Dslice(np.concatenate([img,img_final],axis=1))

































"""分patch参考如下代码"""
# a = np.array(list(range(20)))
# for i in range(1, a.shape[0], 2):
#     # try:
#     #     print(a[i:i+7:1])
#     # except :
#     #     # print(a[i::1])
#     #     print('s',i)
#     #     break
#     print(a[i:i + 7:1]) if len(a[i:i + 7:1])==7

# 风车的操作表较难，所以还原回去，需要再旋转，才能和网格对齐，不过这样可能会损失，，一些信息
# 另外风车的角度间隔也要限制一下，不要太大，否则很诡异









# 还原patch的时候，额外建立一个array，记录同一个voxel被赋值的次数，最后取平均就完事了

# 还原可能会用到的操作
#     a = np.array([0,0,0,0,0,0,0,0,0])
#     a[1:10:2] = np.array([1,2,3,4])




