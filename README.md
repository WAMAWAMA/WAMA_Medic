
# ω👁м👁 medic
A simple-to-use yet function-rich medical image processing toolbox

Highlights: 
1) 2D and 3D visualization directly in the python environment, which is convenient for debugging code; 
2) provides functions such as resampleing, dividing patches, restoring patches, etc.





**Environmental Preparation**
 - `simpleITK`
 - [`batchgenerator`](https://github.com/MIC-DKFZ/batchgenerators)
 - `mayavi`（optional，cannot be installed directly on Windows, please search for the installation method by yourself, or directly copy the `conda env` of the installed mayavi in the following download link to the envs path of the local conda.）
 
 mayavi conda env: [[baidu disk with pw:tqu4]](https://pan.baidu.com/s/1DsddpkbWJ9vexi94xv2dXA)
 
 demo data: [[Google drive]](https://drive.google.com/drive/folders/17Gq9TaU057ptgc5PIFrOy5jmnxRRktCW?usp=sharing)（Includes CT and MRI from TotalSegmentator and MICCAI 2015 OAR datasets, respectively）

# Main function

  - Load medical images in nii or nii.gz format (one patient can load multiple modalities)
  - Voxel resampling
  - 2D, 3D, nD image scaling
  - Get the bounding box of the mask
  - Window width and window level adjustment
  - Arbitrary dimension split or **reorganize** patches
  - 3D volume visualization of original image, mask, bbox (interactive)
  - 3D layer visualization of original image, mask, bbox (interactive)
  - Generate bbox (ie ROI) according to the mask, and crop the image in the ROI


can be used to
  - Data preprocessing such as resampling
  - extract patches from scan
  - reorganize scan from patches
  - Observe the overall effect after 3D amplification (such as 3D warping and patch recombination)



todo
  - [ ] Visual transparency control
  - [ ] Multi-class segmentation label visualization
  - [ ] Registration algorithm between cases and modes
  - [ ] Optimize processing speed
  - [ ] Derivative images such as edge enhancement, wavelet decomposition

## 1.加载原始图像和mask,体素重采样，调整窗宽窗位，3D可视化


```python

from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # 构建实例
subject1.appendImageFromNifti('CT', img_path)  # 加载图像，自定义模态名，如‘CT’
subject1.appendSementicMaskFromNifti('CT', mask_path)  # 加载mask，注意模态名要对应
# 也可以使用appendImageAndSementicMaskFromNifti同时加载图像和mask

subject1.resample('CT', aim_spacing=[1, 1, 1])  # 重采样到1x1x1 mm， 注意单位是mm
subject1.adjst_Window('CT', WW = 321, WL = 123) # 调整窗宽窗位

# 可视化
subject1.show_scan('CT', show_type='slice')  # 显示原图，slice模式
subject1.show_scan('CT', show_type='volume')  # 显示原图，volume模式

subject1.show_MaskAndScan('CT', show_type='volume') # 同时显示原图和mask


subject1.show_bbox('CT', 2)  # 显示bbox形状，注意，不存在bbox时，自动从mask生成最小外接矩阵为bbox

```


<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scan_slicemode.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scan_volumemode.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>显示原图，slice模式</th>
<th>显示原图，volume模式</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scanandmask_volumemode.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_bbox_volumemode.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>同时显示原图和mask</th>
<th>显示bbox形状</th>
</tr>

</table>


## 2.任意维度分割或重组patch

准确的说，是将patch还原到原始空间对应的位置
如一个patch经过分割网络，输出该patch的分割结果，一个即可还原到原始位置可视化。


```python
from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # 构建实例
subject1.appendImageFromNifti('CT', img_path)  # 加载图像，自定义模态名，如‘CT’
subject1.appendSementicMaskFromNifti('CT', mask_path)  # 加载mask，注意模态名要对应
# 也可以使用appendImageAndSementicMaskFromNifti同时加载图像和mask

subject1.resample('CT', aim_spacing=[1, 1, 1])  # 重采样到1x1x1 mm， 注意单位是mm
subject1.adjst_Window('CT', WW=321, WL=123)  # 调整窗宽窗位

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




```


<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_all_z.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_half_z.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>设置一：沿着Z轴分patch，并放回所有patch</th>
<th>设置一：沿着Z轴分patch，并放回一半patch</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_squared.gif"  height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/3_show_patches_expand.gif"  height="283" width="324" alt="input images"></td>
</tr>

<tr>
<th>设置二：分块（类似魔方）</th>
<th>设置三：观察膨胀系数的影响</th>
</tr>

</table>


## 3.图像增强或扩增（3D）


```python

from wama.utils import *
from wama.data_augmentation import aug3D

img_path = r'D:\git\testnini\s22_v1.nii.gz'
mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'

subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square', aim_shape=[128,128,128])


bbox_image_batch = np.expand_dims(np.stack([bbox_image,bbox_image,bbox_image,bbox_image,bbox_image]),axis=1)# 构建batch
bbox_mask_batch = np.zeros(bbox_image_batch.shape)
bbox_mask_batch[:,:,20:100,20:100,20:100] = 1

auger = aug3D(size=[128,128,128], deformation_scale = 0.25) # size为原图大小即可（或batch大小）
aug_result = auger.aug(dict(data=bbox_image_batch, seg = bbox_mask_batch))  # 注意要以字典形式传入

# 可视化
index = 1
show3D(np.concatenate([np.squeeze(aug_result['seg'][index],axis=0),np.squeeze(bbox_mask_batch[index],axis=0)],axis=1))
aug_img = np.squeeze(aug_result['data'][index],axis=0)
show3D(np.concatenate([aug_img,bbox_image],axis=1)*100)

```


<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/4_scan_aug.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/4_mask_aug.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>原图，扩增前后</th>
<th>mask，扩增前后</th>
</tr>


</table>


## ?.图像减裁
```python
from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # 构建实例
subject1.appendImageFromNifti('CT', img_path)  # 加载图像，自定义模态名，如‘CT’
subject1.appendSementicMaskFromNifti('CT', mask_path)  # 加载mask，注意模态名要对应
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
```
