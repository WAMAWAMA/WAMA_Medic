import numpy as np
import SimpleITK as sitk
import warnings
from scipy.ndimage import zoom
import pickle
from mayavi import mlab
import os


# 某些库不是必要的，so，若有则导入，否则不需要导入

try:
    from mayavi import mlab
    # from qweqwe import mlab
    print ('mayavi already imported')
    mayavi_exist_flag = True
except:
    print ('no mayavi')
    mayavi_exist_flag = 0





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
    output = output.astype(np.float32)
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



# img_pth = r'D:\git\testnini\s22_v1.nii.gz'
# mask_pth =r'D:\git\testnini\s22_v1_m1.nii.gz'
# img_save_path =r'D:\git\testnini\new_s22_v1.nii.gz'
# mask_save_pth =r'D:\git\testnini\new_s22_v1_m1.nii.gz'
#
#
#
#
# scan,spacing,origin,transfmat,axesOrder = readIMG(img_pth)
# mask,_,_,_,_ = readIMG(mask_pth)
# scan = np.transpose(scan, (2,0,1))
# scan = np.transpose(scan, (2,0,1))
# writeIMG(img_save_path,scan,spacing,origin,transfmat)
#
# from matplotlib import pyplot as plt
# scan = adjustWindow(scan,321,123)
#
# indexx = 200
# plt.subplot(1, 2, 1)
# plt.imshow(scan[:,indexx,:], cmap=plt.gray())
# plt.subplot(1, 2, 2)
# plt.imshow(mask[:,indexx,:], cmap=plt.gray())
# plt.show()


# 自定义error
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



def checkoutIndex(array3D, index):
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

    :param bbox: 坐标 [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    :param trans_rate: (dim1r,dim2r,dim3r)
    :return:
    """
    trans_rate = list(trans_rate)
    trans_rate = [trans_rate[0], trans_rate[0], trans_rate[1],trans_rate[1],trans_rate[2],trans_rate[2]]
    trans_rate = np.array(trans_rate)
    return list(np.array(trans_rate)*np.array(bbox))

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
        目前只支持单肿瘤，抱歉抱歉
        """
        # 可能会用到的一些信息
        self.id = None
        # 存储图像的信息
        self.scan = {}  # 字典形式储存数据，如image['CT']=[1,2,3]， 不同模态的图像必须要是配准的！暂时不支持没配准的
        self.spacing = {}  # 字典形式存储数据的voxelsize，注意，mask不需要这个信息
        self.origin = {}  # 字典形式存储数据的voxelsize，注意，mask不需要这个信息
        self.transfmat = {}  # 字典形式存储数据的voxelsize，注意，mask不需要这个信息
        self.axesOrder = {}  # 字典形式存储数据的voxelsize，注意，mask不需要这个信息

        self.resample_spacing = {}  # 一旦存在，则表示图像已经经过了resample

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
                show3D(self.mask[img_type])
            if show_type == 'slice':
                show3Dslice(self.mask[img_type])
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
        mask = connected_domain_3D(mask)

        # 计算得到bbox，形式为[dim1min, dim1max, dim2min, dim2max, dim3min, dim3max]
        indexx = np.where(mask> 0.)
        dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                                np.min(indexx[1]), np.max(indexx[1]),
                                                                np.min(indexx[2]), np.max(indexx[2])]
        self.bbox_mask[img_type] = [dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]

    def add_box(self, img_type, bbox):
        """
        ！！ 需要在resample操作前进行，一旦经过了resample，就不可以添加Bbox了
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
            self.make_bbox_from_mask()
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
        or_spacing = self.spacing[img_type]
        trans_rate = tuple(np.array(or_spacing)/np.array(aim_spacing))

        # resample， 并记录aim_spacing, 以表示图像是经过resample的
        self.resample_spacing[img_type] = aim_spacing
        # 先对原图操作
        self.scan[img_type] = zoom(self.scan[img_type], trans_rate,order=2) # 用双三次插值？
        # 再对mask操作
        if img_type in self.scan.keys():
            self.sementic_mask[img_type] = zoom(self.sementic_mask[img_type], trans_rate, order=0)  # 最近邻插值？（检查下是不是还是二值图接可）
        # 再对BBox操作（转化为mask，之后resize，之后取bbox）
        if img_type in self.bbox.keys():
            self.bbox[img_type] = bbox_scale(self.bbox(img_type),trans_rate) # todo 需要检查一下



        raise NotImplementedError

    def slice_neibor_add(self, axis, add_num, add_weights):
        """
        任何时候操作都可以？？yes，只能对scan操作
        slice neighbor add, 相邻层累加策略，类似 mitk 里面的那个多层叠加显示的东西
        :return:
        """






        raise NotImplementedError





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
                    axis: list, 滑动的轴，['coronal', 'sagittal', 'axial'], or ['sagittal', 'axial'] or ['x','y','z'] or ['dim1', 'dim2', dim3'] or [0, 1, 2]
                                这里需要注意，有如下对应关系：
                                当滑动轴为1个时，window的size必须有两个维度是与原图尺寸相同的，即原图[80,80,80],size其中两维至少为80，这样才能保证在一个axis滑动
                                同理，滑动轴为2个时，必须size中一维为最大值
                                滑动轴为3个时，size任意一维度都不要大于最大值
                    stride: list， 指定几个滑动轴，就有几个步长

                    返回一个list， 所有patch的坐标列表，坐标形式和scan的axis[x,y,z]对应
                    还会返回patch的数量
                    以上都会储存在类里

                @ windmill 对应的参数 (一般使用这个操作，不需要辅助slice_nb_add这个操作)
                    axis：滑动的轴，只可指定一个，['coronal', 'sagittal', 'axial'], or ['sagittal', 'axial'] or ['x','y','z'] or ['dim1', 'dim2', dim3'] or [0, 1, 2]
                    add_num: 相邻层累加到当前层的厚度，指定为0则不累积，指定为1则当前层n会加上相邻n-1和n+1层的信息
                    add_weights: list，累加时候相邻层的权重，不设置则直接平均，例n为3时，可设置为[3,2,1,1],对应[n,n+1,n+2,n+3]的权重
                    patch_slice: 即patch的厚度，默认为1

        """
        raise NotImplementedError





img_pth = r'D:\git\testnini\s22_v1.nii.gz'
mask_pth =r'D:\git\testnini\s22_v1_m1.nii.gz'
img_save_path =r'D:\git\testnini\new_s22_v1.nii.gz'
mask_save_pth =r'D:\git\testnini\new_s22_v1_m1.nii.gz'


mask = sitk.ReadImage(mask_pth)
mask = sitk.GetArrayFromImage(mask)

mask = connected_domain_3D(mask)
show3D(1-mask.astype(np.float32))
show3D2(mask.astype(np.float32))


patient1 = wama()
patient1.appendImageFromNifti('CT', img_pth, printflag=True)
patient1.appendImageFromNifti('CT_V', img_pth, printflag=True)
patient1.appendImageFromNifti('CT', img_pth, printflag=True)
patient1.appendSementicMaskFromNifti('CT', mask_pth)
# patient1.appendSementicMaskFromNifti('CT_V', r'E:\@data_dasheng_rna\nii_gz_data\arterial\65.nii.gz')
patient1.appendImageAndSementicMaskFromNifti('MRI', img_path=img_pth,mask_path=mask_pth, printflag=True)


# # 序列化到文件
# with open(r"F:\pycodes\ML\a.txt", "wb") as f:
#     pickle.dump(obj, f)
#
# with open(r"F:\\pycodes\\ML\\a.txt", "rb") as f:
#     print(pickle.load(f))# 输出：(123, 'abcdef', ['ac', 123], {'key': 'value', 'key1': 'value1'})




class testobj():
    def __init__(self):
        self.












import scipy.io as scio

path = r'D:\software\wechat\savefile\WeChat Files\wozuiaipopo520\FileStorage\File\2020-09\mwp100100001.mat'
data = scio.loadmat(path)
image = data['data']
image = resize3D(image, [121,145,121])


maskpth = r'D:\software\wechat\savefile\WeChat Files\wozuiaipopo520\FileStorage\File\2020-09\aal.nii'
mask = sitk.ReadImage(maskpth)
mask = sitk.GetArrayFromImage(mask)
mask = mask>0
mask = mask.astype(np.float32)


image = mask*image
# image[image<0.01] = 0.


# img_pth = r'D:\git\testnini\s22_v1_m1.nii.gz'
import SimpleITK as sitk                            #path为文件的路径
# image = sitk.ReadImage(img_pth)
# image = sitk.GetArrayFromImage(image)
# image = adjustWindow(image,321,123)
# image = image/321
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction,PiecewiseFunction
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image), name='3-d ultrasound ')
# ctf = ColorTransferFunction()                       # 该函数决定体绘制的颜色、灰度等
# # for gray_v in range(256):
# #     ctf.add_rgb_point(value, r, g, b)
# vol._volume_property.set_color(ctf)                 #进行更改，体绘制的colormap及color
# vol._ctf = ctf
# vol.update_ctf = True
# otf = PiecewiseFunction()
# otf.add_point(20, 0.1)
# vol._otf = otf
# vol._volume_property.set_scalar_opacity(otf)
# # Also, it might be useful to change the range of the ctf::
# ctf.range = [0, 1]
mlab.vectorbar()
mlab.show()
# fig_myv = mlab.figure(size=(220,220),bgcolor=(1,1,1))
# f = mlab.gcf()
# f.scene._lift()
# frame = mlab.screenshot(antialiased=True)
#
# from matplotlib import pyplot as plt
# plt.imshow(frame)
# plt.show()



mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
                   plane_orientation='x_axes', slice_index=10)         # 设定x轴切面
mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
                   plane_orientation='y_axes', slice_index=10)         # 设定y轴切面
mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
                  plane_orientation='z_axes', slice_index=10)          # 设定z轴切面
mlab.show()


















