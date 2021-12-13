import SimpleITK as sitk
import numpy as np
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
    scan = np.transpose(scan, (2, 0, 1)) #  把顺序该回去

    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)
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
    spacing = itkimage.GetSpacing()        #voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    axesOrder = ['coronal', 'sagittal', 'axial']  # 调整顺序可以直接axesOrder = [axesOrder[0],axesOrder[2],axesOrder[1]]
    return scan,spacing,origin,transfmat,axesOrder
path1 = r'D:\new\1.nii'
metadata = readIMG(path1)
metadata[0][metadata[0]>0] = 1
metadata[0][metadata[0]<=0] = 0
writeIMG(r'D:\new\newnew\1.nii', metadata[0],metadata[1],metadata[2],metadata[3])


for case in dataset_test['train_set']:
	print(case['mask_path'])
	show3Dslice(np.concatenate([mat2gray(case['img']),case['mask']],axis=1))
	f = input()
	if f == '0':
		break