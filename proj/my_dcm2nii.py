"""
参考：https://blog.csdn.net/JianJuly/article/details/81214408
"""
import SimpleITK as sitk
import os
sep = os.sep

def dcm2nii(file_path):
    # Dicom序列所在文件夹路径（在我们的实验中，该文件夹下有多个dcm序列，混合在一起）
    # file_path = r'E:\@data_hcc_rna_mengqi\new\human_FCM\01\ADC'
    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)
    # print(nb_series)

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[1]代表的是第二个序列的ID
    # 如果不添加series_IDs[1]这个参数，则默认获取第一个序列的所有切片路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 将image转换为scan
    # 查看该3D图像的尺寸
    print(image3D.GetSize())
    sitk.WriteImage(image3D, file_path+sep+'nifti.nii.gz')
    print('save succed')





path_all = r'E:\@data_hcc_rna_mengqi\new\mice_rna_MRI'
dir_list = os.listdir(path_all)
dir_list = [path_all+sep+i for i in dir_list if '.' not in i]
dir_list_final = []
for dir in dir_list:
    tmp_dir_list = os.listdir(dir)
    tmp_dir_list = [dir+sep+i for i in tmp_dir_list if '.' not in i]
    for i in tmp_dir_list:
        dir_list_final.append(i)

fail_list = []
for index,i in enumerate(dir_list_final):
    print(i,index+1,'/',len(dir_list_final))
    if not os.path.exists(i+sep+'nifti.nii.gz'):
        try:
            dcm2nii(i)
        except:
            fail_list.append(i)
    else:
        print('resampling:',i)


# 保存未成功转换的图像





