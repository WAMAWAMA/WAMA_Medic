import csv
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage import measure
import copy
sep = os.sep

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def writeCsv(csfname,rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)
        
def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan, spacing, origin, transfmat

def writeMhd(filename,scan,spacing,origin,transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)    

def getImgWorldTransfMats(spacing, transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
    for d in range(3):
        transfmat[0:3, d] = transfmat[0:3,d]*spacing[d]
    transfmat_toworld = transfmat #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
    
    return transfmat_toimg, transfmat_toworld

def convertToImgCoord(xyz,origin,transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg,xyz))    
    return xyz
    
def convertToWorldCoord(xyz,origin,transfmat_toworld):
    # convert image to world coordinates
    xyz = np.matmul(transfmat_toworld,xyz)
    xyz = xyz + origin
    return xyz

def extractCube(scan,spacing,xyz,cube_size=80,cube_size_mm=51):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2,1,0]],np.int)
    spacing = np.array([spacing[i] for i in [2,1,0]])
    scan_halfcube_size = np.array(cube_size_mm/spacing/2,np.int)
    if np.any(xyz<scan_halfcube_size) or np.any(xyz+scan_halfcube_size>scan.shape): # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan,((maxsize,maxsize,)),'constant',constant_values=0)
        xyz = xyz+maxsize
    
    scancube = scan[xyz[0]-scan_halfcube_size[0]:xyz[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    xyz[1]-scan_halfcube_size[1]:xyz[1]+scan_halfcube_size[1],
                    xyz[2]-scan_halfcube_size[2]:xyz[2]+scan_halfcube_size[2]]

    sh = scancube.shape
    scancube = zoom(scancube,(cube_size/sh[0],cube_size/sh[1],cube_size/sh[2]),order=2) #resample for cube_size
    
    return scancube

def get_filelist_frompath(filepath, expname, sample_id=None):
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('_')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def save_itk(image, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def combine_mask(mask1, mask2):
    """
    合并mask的函数，重叠mask取交集，不重叠的直接叠加
    :param mask1:
    :param mask2:
    :return: 合并后的mask，如果有重叠的结节，那么就取最小的那个mask
    """

    # 大概思路是：遍历1连通域，如果某结节与mask2有交集，那么相乘就是该结节combine的结果，再把相乘结果纳入
    # 如果没有交集，那么就直接把该结节纳入即可

    # 各自求连通域
    [mask1_labels, mask1_num] = measure.label(mask1,connectivity=None,return_num=True)
    [mask2_labels, mask2_num] = measure.label(mask2,connectivity=None,return_num=True)

    # 建立个0阵
    final_mask = copy.deepcopy(mask1)
    final_mask = final_mask*0

    # 先对1操作
    for i in range(mask1_num):
        # 先把一个结节的mask拿出来
        tmp_mask = copy.deepcopy(mask1)
        tmp_mask = tmp_mask * 0
        tmp_mask[mask1_labels==i+1]=1
        # 求交集
        cheng_mask = tmp_mask*mask2
        cheng_mask[cheng_mask > 0] = 1
        # 判断
        if len(np.unique(cheng_mask)) > 1:
            final_mask = final_mask + cheng_mask
        else:
            final_mask = final_mask + tmp_mask


    # 再对2操作
    for i in range(mask2_num):
        # 先把一个结节的mask拿出来
        tmp_mask = copy.deepcopy(mask2)
        tmp_mask = tmp_mask * 0
        tmp_mask[mask2_labels==i+1]=1

        cheng_mask = tmp_mask*mask1
        cheng_mask[cheng_mask > 0] = 1

        if len(np.unique(cheng_mask)) > 1:
            final_mask = final_mask + cheng_mask
        else:
            final_mask = final_mask + tmp_mask

    final_mask[final_mask > 0] = 1

    return final_mask


def preprocess_mask(mask, finding_id, finding_class, finding_volume):
    """

    :param mask: 输入mask
    :param arg:对应mask的CT的信息，用来区分对应区域是否为结节等
    :return:
    """
    # 对mask进行处理，分离成大于3mm结节mask，小于3mm结节mask，以及非结节的mask
    final_mask_over3 = copy.deepcopy(mask)
    final_mask_over3 = final_mask_over3*0
    final_mask_less3 = copy.deepcopy(final_mask_over3)
    final_mask_nonudel = copy.deepcopy(final_mask_over3)

    for index, nodule in enumerate(finding_id):
        print(nodule,index,finding_volume[index],finding_class[index])
        # 如果是非结节，那么就保存到非结节的mask中去
        if finding_class[index] == 0:
            final_mask_nonudel[mask==nodule] = 1
            print('non nodule')
        # 如果小于3mm结节，那么就保存3mm结节mask中
        elif finding_class[index] == 1 and finding_volume[index]<14.14:
            final_mask_less3[mask == nodule] = 1
            print('less 3mm')
        else:
            final_mask_over3[mask == nodule] = 1
            print('more 3mm')
    return [final_mask_over3, final_mask_less3, final_mask_nonudel]


def get_nodel_detail(LNDbID, rad_id, N_nodules, N_header):
    finding_id = []
    finding_class = []
    finding_volume = []
    for nn in N_nodules:
        if int(nn[N_header.index('LNDbID')]) == LNDbID and int(nn[N_header.index('RadID')]) == rad_id:
            finding_id.append(int(nn[N_header.index('FindingID')]))
            finding_class.append(int(nn[N_header.index('Nodule')]))
            finding_volume.append(float(nn[N_header.index('Volume')]))

    return [finding_id, finding_class, finding_volume]




if __name__ == "__main__":
    # 设置路径
    path_CT = r'G:\@challenge_lungnoduls\@challenge_LNDb\@data\@datas'
    path_mask = r'G:\@challenge_lungnoduls\@challenge_LNDb\@data\@masks'
    path_trainCTs = r'G:\@challenge_lungnoduls\@challenge_LNDb\@data\@trainset_csv\trainCTs.csv'  # 判断一个CT有几个医生annotate过，即一个CT对应有几个mask
    path_nodel_raw = r'G:\@challenge_lungnoduls\@challenge_LNDb\@data\@trainset_csv\trainNodules.csv'  # 每一个病变的细节文件
    save_path = r'G:\@challenge_lungnoduls\@challenge_LNDb\@data\@data_new'

    # 读取文件
    CT_list = get_filelist_frompath(path_CT, 'mhd')
    CTcsvlines = readCsv(path_trainCTs)
    header = CTcsvlines[0]
    nodules = CTcsvlines[1:]

    Nudelcsvlines = readCsv(path_nodel_raw)
    N_header = Nudelcsvlines[0]
    N_nodules = Nudelcsvlines[1:]

    # 批处理
    for indexx, file in enumerate(CT_list):
        print('doing with ', file,'(',indexx+1,'/',len(CT_list),')')
        file_name = file.split(sep)[-1]
        LNDbID = int((file_name.split('-')[-1]).split('.')[0])
        # load & save CT -------------------------------------------------------------------------------------------------
        [scan, spacing, origin, _] = readMhd(file)
        save_itk(scan, origin, spacing, save_path+sep+str(LNDbID)+'.nii.gz')

        # load & save mask after merging -----------------------------------------------------------------------------------
        # find how many rad who annotate the CT
        for n in nodules:
            if int(n[header.index('LNDbID')]) == LNDbID:
                RadN = int(n[header.index('RadN')])
                break
        print(LNDbID, ' have ', RadN, ' RadNs annotation')

        # firstly, save the first rad mask
        # （大于3mm结节，小于3mm结节和非结节病变分别保存，先读取第一个，然后后面的逐渐合并到第一个上）
        rad_id = 1
        maskpath = path_mask+sep+'LNDb-{:04}_rad{}.mhd'.format(LNDbID,rad_id)
        [scan, spacing, origin, _] = readMhd(maskpath)
        # 接下来找到对应CT（LNDbID）的对应医生（rad_id）标注的各个结节的信息，储存在xx结构中传给preprocess_mask
        [finding_id, finding_class, finding_volume]=get_nodel_detail(LNDbID, rad_id, N_nodules, N_header)
        # 预处理（即把不同mask分开），然后分别保存
        final_mask_over3, final_mask_less3, final_mask_nonudel = preprocess_mask(scan, finding_id, finding_class, finding_volume)
        # 保存
        # save_itk(final_mask_over3, origin, spacing, save_path+sep+str(LNDbID)+'_'+str(rad_id)+'over.nii.gz')
        # save_itk(final_mask_less3, origin, spacing, save_path+sep+str(LNDbID)+'_'+str(rad_id)+'less.nii.gz')
        # save_itk(final_mask_nonudel, origin, spacing, save_path+sep+str(LNDbID)+'_'+str(rad_id)+'non.nii.gz')
        # loop for saving others' mask and combining all mask into one array（处理剩余几个rad的部分）
        for i in range(RadN-1):
            rad_id += 1
            maskpath = path_mask+sep+'LNDb-{:04}_rad{}.mhd'.format(LNDbID,rad_id)
            [tmp_scan, spacing, origin, _] = readMhd(maskpath)
            [finding_id, finding_class, finding_volume] = get_nodel_detail(LNDbID, rad_id, N_nodules, N_header)
            tmp_final_mask_over3, tmp_final_mask_less3, tmp_final_mask_nonudel = preprocess_mask(tmp_scan, finding_id, finding_class, finding_volume)
            # save_itk(tmp_final_mask_over3, origin, spacing, save_path + sep + str(LNDbID) + '_' + str(rad_id) + 'over.nii.gz')
            # save_itk(tmp_final_mask_less3, origin, spacing, save_path + sep + str(LNDbID) + '_' + str(rad_id) + 'less.nii.gz')
            # save_itk(tmp_final_mask_nonudel, origin, spacing, save_path + sep + str(LNDbID) + '_' + str(rad_id) + 'non.nii.gz')

            # scan = combine_mask(scan,tmp_scan)
            final_mask_over3 = combine_mask(final_mask_over3, tmp_final_mask_over3)
            final_mask_less3 = combine_mask(final_mask_less3, tmp_final_mask_less3)
            final_mask_nonudel = combine_mask(final_mask_nonudel,tmp_final_mask_nonudel)

        # finaly, save final combined mask
        # 保存
        # save_itk(final_mask_over3, origin, spacing, save_path+sep+str(LNDbID)+'_all_over.nii.gz')
        # save_itk(final_mask_less3, origin, spacing, save_path+sep+str(LNDbID)+'_all_less.nii.gz')
        # save_itk(final_mask_nonudel, origin, spacing, save_path+sep+str(LNDbID)+'_all_non.nii.gz')
        final_mask_all = final_mask_over3+final_mask_less3+final_mask_nonudel
        final_mask_all[final_mask_all > 0] = 1
        save_itk(final_mask_all, origin, spacing, save_path+sep+str(LNDbID)+'_all.nii.gz')









