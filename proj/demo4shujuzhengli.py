# 设置路径
img = r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\data\a.nii'  # 图像， .nii或.nii.gz都可以，没有就填None
mask = r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\data\a_mask.nii'  # 分割（注意要和图像对应同一期象） .nii或.nii.gz都可以，没有就填None
# mask = r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\data\Untitled.nii.gz'  # 分割（注意要和图像对应同一期象） .nii或.nii.gz都可以，没有就填None
landmark = r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\data\a_landmark.nii'  # 粗标注（注意要和图像对应同一期象） .nii或.nii.gz都可以，没有就填None
# 注意，一定不能将新图片保存在原图片的同一文件夹！！！！！
save_folder = r'E:\@data_NENs\@data_medai_format_v1.0.0\data\290129_wangxuejuan\data\newdata'






# 首先需要安装 SimpleITK 库
import SimpleITK as sitk
import numpy as np
import os

sep = os.sep
def mkf(pth):
    try:
        os.makedirs(pth)
    except:
        pass


mkf(save_folder)


def readIMG(filename):
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage)
    spacing = itkimage.GetSpacing()
    origin = itkimage.GetOrigin()
    transfmat = itkimage.GetDirection()
    return scan, spacing, origin, transfmat


def writeIMG(filename, scan, spacing, origin, transfmat):
    itkim = sitk.GetImageFromArray(scan)
    itkim.SetSpacing(spacing)
    itkim.SetOrigin(origin)
    itkim.SetDirection(transfmat)
    sitk.WriteImage(itkim, filename, False)


def connected_domain_3D(image):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image)
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    return labeled_img.astype(np.uint16)


def trans1(landmark_scan, mask_scan_instance):
    label_lm_image = connected_domain_3D(landmark_scan)
    for i in range(len(np.unique(label_lm_image)) - 1):
        landmark_scan[label_lm_image == i + 1] = np.max(mask_scan_instance[label_lm_image == i + 1])
    return landmark_scan


if img is not None:
    if img.split('.')[-1] == 'gz':
        new_img_pth = save_folder + sep + img.split(sep)[-1]
    else:
        new_img_pth = save_folder + sep + img.split(sep)[-1] + '.gz'

    img_scan, spacing, origin, transfmat = readIMG(img)
    writeIMG(new_img_pth, img_scan, spacing, origin, transfmat)

if mask is not None:
    if mask.split('.')[-1] == 'gz':
        new_mask_pth = save_folder + sep + mask.split(sep)[-1]
    else:
        new_mask_pth = save_folder + sep + mask.split(sep)[-1] + '.gz'

    mask_scan, spacing, origin, transfmat = readIMG(mask)
    if len(np.unique(mask_scan)) == 2:  # 如果不是实例标注，则转换
        mask_scan_instance = connected_domain_3D(mask_scan)
    else:
        mask_scan_instance = mask_scan

    writeIMG(new_mask_pth, mask_scan_instance, spacing, origin, transfmat)

# 必须要有mask才能处理
if landmark is not None and mask is not None:
    if landmark.split('.')[-1] == 'gz':
        new_landmark_pth = save_folder + sep + landmark.split(sep)[-1]
    else:
        new_landmark_pth = save_folder + sep + landmark.split(sep)[-1] + '.gz'

    landmark_scan, spacing, origin, transfmat = readIMG(landmark)
    if len(np.unique(landmark_scan)) != 2:  # 如果不是实例标注，则转换
        landmark_scan_instance = landmark_scan
    else:
        landmark_scan_instance = trans1(landmark_scan, mask_scan_instance)

    writeIMG(new_landmark_pth, landmark_scan_instance, spacing, origin, transfmat)
