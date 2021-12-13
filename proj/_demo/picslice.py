# 有的mask是有时间轴的，需要去掉
import numpy as np
import SimpleITK as sitk
import imageio
from scipy.ndimage import zoom
import pickle
import os
import math
import matplotlib.pyplot as plt
from copy import deepcopy


filename = r"D:\software\wechat\savefile\WeChat Files\wozuiaipopo520\FileStorage\File\2021-06\qqwe\+C\ZhangXiaoZhen_801_1195_0439.dcm.nii.gz"

itkimage = sitk.ReadImage(filename)
# 读取图像数据
scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
# 读取图像信息
spacing = itkimage.GetSpacing()  # voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
origin = itkimage.GetOrigin()  # world coordinates of origin
transfmat = itkimage.GetDirection()  # 3D rotation matrix

scan = scan[0]
origin = origin[:-1]
spacing = spacing[:-1]
transfmat_new = (transfmat[0],
                 transfmat[1],
                 transfmat[2],

                 transfmat[4],
                 transfmat[5],
                 transfmat[6],

                 transfmat[8],
                 transfmat[9],
                 transfmat[10]
                 )
transfmat = transfmat_new


itkim = sitk.GetImageFromArray(scan, isVector=False)  # 3D image
itkim.SetSpacing(spacing)  # voxelsize
itkim.SetOrigin(origin)  # world coordinates of origin
itkim.SetDirection(transfmat)  # 3D rotation matrix
sitk.WriteImage(itkim, filename, False)


