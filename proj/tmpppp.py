import numpy as np
import SimpleITK as sitk
import os
sep = os.sep
import json
import warnings


imgpth = r''
savepth = r''

itkimage = sitk.ReadImage(imgpth)
# 读取图像数据
scan = sitk.GetArrayFromImage(itkimage)  # 3D image, 对应的axis[axial,coronal,sagittal], 我们这里当作[z，y，x]
spacing = itkimage.GetSpacing()  # voxelsize，对应的axis[sagittal,coronal,axial]，即[x, y, z]  已确认
origin = itkimage.GetOrigin()  # world coordinates of origin
transfmat = itkimage.GetDirection()  # 3D rotation matrix


scan[scan == 2] = 0

# 覆盖储存
itkim = sitk.GetImageFromArray(scan, isVector=False)  # 3D image
itkim.SetSpacing(spacing)  # voxelsize
itkim.SetOrigin(origin)  # world coordinates of origin
itkim.SetDirection(transfmat)  # 3D rotation matrix

sitk.WriteImage(itkim, savepth, False)