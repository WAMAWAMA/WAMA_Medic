from wama.utils import *


#
img_path = r'D:\git\testnini\s22_v1.nii.gz'
mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'
# img_path = r'D:\git\testnini\s22_v1.nii.gz'
# mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'
subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)
bbox = subject1.getBbox('CT')
# 平滑
# tmpimage = subject1.slice_neibor_add('CT',axis=[2],add_num=[21],add_weights='Mean')
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square')
bbox_mask = subject1.getMaskfromBbox('CT',ex_mode='square')
# mask_image = subject1.getImagefromMask('CT')
# bbox_mask = subject1.getMaskfromBbox('CT',ex_mode='square')

# show3D(mask_image)
# show3Dslice(subject1.scan['CT'])
# show3Dslice(tmpimage)
# show3Dslice(bbox_image)
# patches = slide_window_one_axis(bbox_image,
#                            spacing = None,
#                            origin=None,
#                            transfmat=None,
#                            axesOrder=None,
#                            bbox = [0, bbox_image.shape[0],0, bbox_image.shape[1],0, bbox_image.shape[2]],
#                            axis = 2,
#                            slices = 7,
#                            stride = 5,
#                            expand_r = 3,
#                            mask = None,
#                            ex_mode = 'bbox',
#                            ex_voxels = 0,
#                            ex_mms = None,
#                            resample_spacing=None,
#                            aim_shape = None)
#
#
# reconstuct_img = slide_window_one_axis_reconstruct(patches)
patches = slide_window_n_axis(bbox_image,
                           spacing = None,
                           origin=None,
                           transfmat=None,
                           axesOrder=None,
                           bbox = [0, bbox_image.shape[0],0, bbox_image.shape[1],0, bbox_image.shape[2]],
                           slices = [bbox_image.shape[0]//4-4,bbox_image.shape[1]//2-4,bbox_image.shape[2]//2-4],
                           stride = [20,20,20],
                           expand_r = [1,1,1],
                           mask = bbox_mask,
                           ex_mode = 'bbox',
                           ex_voxels = [0,0,0],
                           ex_mms = None,
                           resample_spacing=None,
                           aim_shape = None)

# for pp in patches:
#     pp.data = pp.mask
# reconstuct_img = slide_window_n_axis_reconstruct([patches[0]])
reconstuct_img = slide_window_n_axis_reconstruct(patches)
# show3Dslice(mat2gray(np.concatenate([reconstuct_img,bbox_image],axis=1)))
show3D(np.concatenate([reconstuct_img,bbox_image],axis=1))
# show3D((reconstuct_img))
# show3D((bbox_image))
# show3Dslice(bbox_image)
# show3Dslice(bbox_image)

# bbox = subject1.getBbox('CT')
#
#
# show3Dslice(bbox_image)
# show3Dslice(mask_image)
#
#
# scan,spacing,_,_,axesOrder = readIMG(img_pth)
# # mask,_,_,_,_ = readIMG(mask_pth)
# scan = adjustWindow(scan,321,123)
#
#
#
#
# # show3Dslice(np.concatenate([mat2gray(mask),mat2gray(scan)],axis=1))
# # show3D(scan)
#
# # 保存数据
# import pickle
# data_output = open('data.pkl','wb')
# pickle.dump(scan,data_output)
# data_output.close()
#
# # 读取数据
# data_input = open('data.pkl','rb')
# read_data = pickle.load(data_input)
# data_input.close()
#
#
# scan_2 = slice_neibor_add_one_dim(scan, 'z', 21, 'Gaussian', 10)
# scan_2 = slice_neibor_add_one_dim(scan_2, 'x', 21, 'Mean', 1)
# scan_2 = slice_neibor_add_one_dim(scan_2, 'y', 21, 'Mean', 1)
# # show3D(scan_2)
# show3Dslice(np.concatenate([scan,scan_2],axis=1))
#
# indexx = -2
# # show2D(np.concatenate([scan_2[:,:,indexx],scan[:,:,indexx]],axis=1))
# show2D(np.concatenate([scan_2[:,:,indexx],scan[:,:,indexx]],axis=1))
# show2D([:,:,0])
#
#
#
# mask_pth =r'D:\git\testnini\s22_v1_m1.nii.gz'
# img_save_path =r'D:\git\testnini\new_s22_v1.nii.gz'
# mask_save_pth =r'D:\git\testnini\new_s22_v1_m1.nii.gz'
#
#
# mask = sitk.ReadImage(mask_pth)
# mask = sitk.GetArrayFromImage(mask)
#
# mask = connected_domain_3D(mask)
# show3D(1-mask.astype(np.float32))
# show3D2(mask.astype(np.float32))
#
#
# patient1 = wama()
# patient1.appendImageFromNifti('CT', img_pth, printflag=True)
# patient1.appendImageFromNifti('CT_V', img_pth, printflag=True)
# patient1.appendImageFromNifti('CT', img_pth, printflag=True)
# patient1.appendSementicMaskFromNifti('CT', mask_pth)
# # patient1.appendSementicMaskFromNifti('CT_V', r'E:\@data_dasheng_rna\nii_gz_data\arterial\65.nii.gz')
# patient1.appendImageAndSementicMaskFromNifti('MRI', img_path=img_pth,mask_path=mask_pth, printflag=True)
#
#
# # # 序列化到文件
# # with open(r"F:\pycodes\ML\a.txt", "wb") as f:
# #     pickle.dump(obj, f)
# #
# # with open(r"F:\\pycodes\\ML\\a.txt", "rb") as f:
# #     print(pickle.load(f))# 输出：(123, 'abcdef', ['ac', 123], {'key': 'value', 'key1': 'value1'})
#
#
#
#
# class testobj():
#     def __init__(self):
#         self.
#
#
#
#
#
#
#
#
#
#
#
#
# import scipy.io as scio
#
# path = r'D:\software\wechat\savefile\WeChat Files\wozuiaipopo520\FileStorage\File\2020-09\mwp100100001.mat'
# data = scio.loadmat(path)
# image = data['data']
# image = resize3D(image, [121,145,121])
#
#
# maskpth = r'D:\software\wechat\savefile\WeChat Files\wozuiaipopo520\FileStorage\File\2020-09\aal.nii'
# mask = sitk.ReadImage(maskpth)
# mask = sitk.GetArrayFromImage(mask)
# mask = mask>0
# mask = mask.astype(np.float32)
#
#
# image = mask*image
# # image[image<0.01] = 0.
#
#
# # img_pth = r'D:\git\testnini\s22_v1_m1.nii.gz'
# import SimpleITK as sitk                            #path为文件的路径
# # image = sitk.ReadImage(img_pth)
# # image = sitk.GetArrayFromImage(image)
# # image = adjustWindow(image,321,123)
# # image = image/321
# from mayavi import mlab
# from tvtk.util.ctf import ColorTransferFunction,PiecewiseFunction
# vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image), name='3-d ultrasound ')
# # ctf = ColorTransferFunction()                       # 该函数决定体绘制的颜色、灰度等
# # # for gray_v in range(256):
# # #     ctf.add_rgb_point(value, r, g, b)
# # vol._volume_property.set_color(ctf)                 #进行更改，体绘制的colormap及color
# # vol._ctf = ctf
# # vol.update_ctf = True
# # otf = PiecewiseFunction()
# # otf.add_point(20, 0.1)
# # vol._otf = otf
# # vol._volume_property.set_scalar_opacity(otf)
# # # Also, it might be useful to change the range of the ctf::
# # ctf.range = [0, 1]
# mlab.vectorbar()
# mlab.show()
# # fig_myv = mlab.figure(size=(220,220),bgcolor=(1,1,1))
# # f = mlab.gcf()
# # f.scene._lift()
# # frame = mlab.screenshot(antialiased=True)
# #
# # from matplotlib import pyplot as plt
# # plt.imshow(frame)
# # plt.show()
#
#
#
# mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
#                    plane_orientation='x_axes', slice_index=10)         # 设定x轴切面
# mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
#                    plane_orientation='y_axes', slice_index=10)         # 设定y轴切面
# mlab.volume_slice(image, colormap='gray', extent=[0,117,0,246,0,192],
#                   plane_orientation='z_axes', slice_index=10)          # 设定z轴切面
# mlab.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
