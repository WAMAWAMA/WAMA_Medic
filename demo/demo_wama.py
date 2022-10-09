from wama.utils import *

img_path = r"F:\@data_guoyuan\we\TCGA-BRCA ROI\low-group\TCGA-E2-A1IE\TCGA-E2-A1IE-img.nii.gz"
mask_path = r"D:\git\testnini\heatmap_new.nii.gz"

subject1 = wama()  # 构建实例
subject1.appendImageFromNifti('CT', mask_path)  # 加载图像，自定义模态名，如‘CT’
subject1.appendSementicMaskFromNifti('CT', mask_path)  # 加载mask，注意模态名要对应
subject1.getBbox()
# 也可以使用appendImageAndSementicMaskFromNifti同时加载图像和mask

print(subject1.scan['CT'].shape)


# 截取
subject1.scan['CT'] = subject1.scan['CT'][:,:,:100]
subject1.sementic_mask['CT'] = subject1.sementic_mask['CT'][:,:,:100]

print(subject1.scan['CT'].shape)


writeIMG(r"D:\git\testnini\s1_v1_cut.nii",
		 subject1.scan['CT'],
		 subject1.spacing['CT'],
		 subject1.origin['CT'],
		 subject1.transfmat['CT'])
writeIMG(r"D:\git\testnini\s1_v1_m1_w_cut.nii",
		 subject1.sementic_mask['CT'],
		 subject1.spacing['CT'],
		 subject1.origin['CT'],
		 subject1.transfmat['CT'])









# 平滑去噪
qwe = subject1.slice_neibor_add('CT', axis=['z'], add_num=[7], add_weights='Gaussian')  # 使用高斯核，在z轴平滑

# 提取bbox内图像（bbox即分割mask的最小外接矩阵）
bbox_image = subject1.getImagefromBbox('CT', ex_mode='square', ex_mms=24, aim_shape=[256, 256, 256])

"""
    分patch的逻辑：
    1）先框取ROI获得bbox，之后在ROI内进行操作
    2）外扩roi
    3）将roi内图像拿出，缩放到aim_shape
    4）分patch
"""

# 分patch，设置一：沿着Z轴分patch，patch为2D，且每隔10层取一层
subject1.makePatch(mode='slideWinND',  # 默认的即可
				   img_type='CT',  # 关键字
				   aim_shape=[256, 256, 256],  # 缩放到这个尺寸
				   slices=[256 // 2, 256 // 2, 1],  # 每个patch在各个维度的长度（注意，Z轴为1，即沿着Z轴分2D patch）
				   stride=[256 // 2, 256 // 2, 10],  # patch在各个轴的滑动步长（注意这里z轴是10）
				   expand_r=[1, 1, 1],  # 类似膨胀卷积（空洞卷积）的膨胀系数，1即不膨胀
				   ex_mode='square',  # 取bbox后，将bbox变为正方体
				   ex_mms=24,  # bbox外扩（或变为正方体后）多少毫米
				   )
reconstuct_img = slide_window_n_axis_reconstruct(subject1.patches['CT'])  # 将patch放回原空间
reconstuct_img_half = slide_window_n_axis_reconstruct(
	subject1.patches['CT'][:len(subject1.patches['CT']) // 2])  # 将一半的patch放回原空间

patch = subject1.patches['CT']  # 获取patch
show3D(np.concatenate([bbox_image, reconstuct_img], axis=1))
show3D(np.concatenate([bbox_image, reconstuct_img_half], axis=1))



# 分patch，设置二：分块（类似魔方）
subject1.makePatch(mode='slideWinND',  # 默认的即可
				   img_type='CT',  # 关键字
				   aim_shape=[256, 256, 256],  # 缩放到这个尺寸
				   slices=[256 // 8, 256 // 8,  256 // 8],  # 每个patch在各个维度的长度（注意，Z轴为1，即沿着Z轴分2D patch）
				   stride=[( 256 // 8)+3, ( 256 // 8)+3, ( 256 // 8)+3],  # patch在各个轴的滑动步长（注意这里z轴是10）
				   expand_r=[1, 1, 1],  # 类似膨胀卷积（空洞卷积）的膨胀系数，1即不膨胀
				   ex_mode='square',  # 取bbox后，将bbox变为正方体
				   ex_mms=24,  # bbox外扩（或变为正方体后）多少毫米
				   )
reconstuct_img = slide_window_n_axis_reconstruct(subject1.patches['CT'])  # 将patch放回原空间
reconstuct_img_half = slide_window_n_axis_reconstruct(
	subject1.patches['CT'][:len(subject1.patches['CT']) // 2])  # 将一半的patch放回原空间

patch = subject1.patches['CT']  # 获取patch
show3D(np.concatenate([bbox_image, reconstuct_img], axis=1))
show3D(np.concatenate([bbox_image, reconstuct_img_half], axis=1))



# 分patch，设置三：观察膨胀系数的影响（其实基本用不到）
subject1.makePatch(mode='slideWinND',  # 默认的即可
				   img_type='CT',  # 关键字
				   aim_shape=[256, 256, 256],  # 缩放到这个尺寸
				   slices=[30, 30, 30],  # 每个patch在各个维度的长度（注意，Z轴为1，即沿着Z轴分2D patch）
				   stride=[1, 1, 1],  # patch在各个轴的滑动步长（注意这里z轴是10）
				   expand_r=[5, 5, 5],  # 类似膨胀卷积（空洞卷积）的膨胀系数，1即不膨胀
				   ex_mode='square',  # 取bbox后，将bbox变为正方体
				   ex_mms=24,  # bbox外扩（或变为正方体后）多少毫米
				   )

reconstuct_img_onlyone = slide_window_n_axis_reconstruct([subject1.patches['CT'][0]])  # 将一个patch放回原空间（观察膨胀系数的影响）

patch = subject1.patches['CT'] # 获取patch
show3D(np.concatenate([bbox_image, reconstuct_img_onlyone], axis=1))
