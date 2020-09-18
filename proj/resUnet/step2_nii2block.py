import csv
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage import measure
import copy
import h5py
sep = os.sep




def get_filelist_frompath4newnii(filepath, expname, sample_id=None):
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
            if file.endswith('.' + expname) and '_' not in file:
                file_List.append(os.path.join(filepath, file))
    return file_List


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat


def getImgWorldTransfMats(spacing, transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3], transfmat[3:6], transfmat[6:9]])
    for d in range(3):
        transfmat[0:3, d] = transfmat[0:3, d] * spacing[d]
    transfmat_toworld = transfmat  # image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat)  # world to image coordinates conversion matrix

    return transfmat_toimg, transfmat_toworld


def convertToImgCoord(xyz, origin, transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg, xyz))
    return xyz


def extractCube(scan, spacing, xyz, cube_size=80, cube_size_mm=51):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2, 1, 0]], np.int)
    spacing = np.array([spacing[i] for i in [2, 1, 0]])
    scan_halfcube_size = np.array(cube_size_mm / spacing / 2, np.int)
    if np.any(xyz < scan_halfcube_size) or np.any(
                            xyz + scan_halfcube_size > scan.shape):  # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan, ((maxsize, maxsize,)), 'constant', constant_values=0)
        xyz = xyz + maxsize

    scancube = scan[xyz[0] - scan_halfcube_size[0]:xyz[0] + scan_halfcube_size[0],  # extract cube from scan at xyz
               xyz[1] - scan_halfcube_size[1]:xyz[1] + scan_halfcube_size[1],
               xyz[2] - scan_halfcube_size[2]:xyz[2] + scan_halfcube_size[2]]

    sh = scancube.shape
    scancube = zoom(scancube, (cube_size / sh[0], cube_size / sh[1], cube_size / sh[2]),
                    order=2)  # resample for cube_size

    return scancube


if __name__ == "__main__":

    filepath = r'/media/root/老王/@data_LNDb/@data_newnew'
    filelist = get_filelist_frompath4newnii(filepath,'gz')
    personnum=len(filelist)

    # 读取结节数据
    path_nodel_gt = r'/media/root/老王/@data_LNDb/LNDb dataset/trainset_csv/trainNodules_gt.csv'
    CTcsvlines = readCsv(path_nodel_gt)
    header = CTcsvlines[0]
    nodules = CTcsvlines[1:]

    path_CT_F = r'/media/root/老王/@data_LNDb/LNDb dataset/trainset_csv/trainFleischner.csv'
    Fcsvlines = readCsv(path_CT_F)
    Fheader = Fcsvlines[0]
    Fnodules = Fcsvlines[1:]


    # savepath:分为3个来储存,大于3mm结节,小于3mm结节,非结节病变
    savepath = r'/media/root/老王/@data_LNDb/@data_80_nii'
    over3_savepath =  savepath+sep+'over3'
    less3_savepath =  savepath+sep+'less3'
    nonnodule_savepath = savepath+sep+'non'
    os.mkdir(over3_savepath)
    os.mkdir(less3_savepath)
    os.mkdir(nonnodule_savepath)




    # 一个一个遍历病人,然后把那个病人所有结节都整出来
    for index, CTfilename  in  enumerate(filelist):
        print('doing with id:',index+1,'/',personnum)
        print(CTfilename)
        # 读取CT
        [scan, spacing, origin, transfmat] = readMhd(CTfilename)
        LNDbID = int((CTfilename.split(sep)[-1]).split('.')[0])
        # 读取mask
        Maskfilename = filepath+sep+str(LNDbID)+'_all.nii.gz'
        [Mscan, spacing, origin, transfmat] = readMhd(Maskfilename)
        # 转换坐标系要用到的
        transfmat_toimg, _ = getImgWorldTransfMats(spacing, transfmat)
        # 分析这个人有几个病灶之类的,以下都是储存结节信息的
        FindingID = []
        ctr = []
        agrlevel = []
        isNodule = []
        isOver3mm = []
        Textlevel = []  # 12为0,3为1,45为2,0为3(因为0textlevel可能用不到)

        for n in nodules:
            if int(n[header.index('LNDbID')]) == LNDbID:
                FindingID.append(int(n[header.index('FindingID')]))
                ctr.append(np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])]))
                agrlevel.append(int(n[header.index('AgrLevel')]))
                isNodule.append(int(n[header.index('Nodule')]))

                volume = float(n[header.index('Volume')])
                if volume<14.14:
                    isOver3mm.append(0)
                else:
                    isOver3mm.append(1)

                texture = int(float(n[header.index('Text')]))
                if texture ==1 or texture==2:
                    Textlevel.append(0)
                elif texture==3:
                    Textlevel.append(1)
                elif texture==4 or texture==5:
                    Textlevel.append(2)
                else:
                    Textlevel.append(3)

        # 加入F评分,但是这个是以人为单位的,所以有点问题
        for n in Fnodules:
            if int(n[Fheader.index('LNDbID')]) == LNDbID:
                F = int(n[Fheader.index('Fleischner')])
                break



        # 读取每个结节,并保存到h5文件中
        for i in range(len(Textlevel)):
            # 计算中心坐标
            real_ctr = convertToImgCoord(ctr[i],origin,transfmat_toimg)
            # 扣出立方提
            scan_cube = extractCube(scan, spacing, real_ctr)
            mask_cube = extractCube(Mscan, spacing, real_ctr)


            if isNodule[i] == 0:
                tmpsavepath = nonnodule_savepath
            else:
                if isOver3mm[i] == 1:
                    tmpsavepath = over3_savepath
                else:
                    tmpsavepath = less3_savepath

            # 保存格式:LNDbID_findingID_agrlevel_isNodule_isover3mm_Textlevel_Filerscore
            # final_save_path = tmpsavepath+\
            #                   sep+str(LNDbID)+\
            #                   '_'+str(FindingID[i])+\
            #                   '_'+str(agrlevel[i])+\
            #                   '_'+str(isNodule[i])+\
            #                   '_'+str(isOver3mm[i])+\
            #                   '_'+str(Textlevel[i])+\
            #                   '_'+str(F)+'.h5'

            final_save_path = tmpsavepath+\
                              sep+str(LNDbID)+\
                              '_'+str(FindingID[i])+\
                              '_'+str(agrlevel[i])+\
                              '_'+str(isNodule[i])+\
                              '_'+str(isOver3mm[i])+\
                              '_'+str(Textlevel[i])+\
                              '_'+str(F)+'.pre.nii.gz'



            itkimage = sitk.GetImageFromArray(scan_cube, isVector=False)
            sitk.WriteImage(itkimage, final_save_path, True)

            # with h5py.File(final_save_path) as f:
            #     f['data'] = scan_cube
            #     f['mask'] = mask_cube





































