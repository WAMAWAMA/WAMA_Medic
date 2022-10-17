
# ωαмα m💊dic
A simple-to-use yet function-rich medical image processing toolbox

Highlights
 - 2D and 3D visualization directly in the python environment, which is convenient for debugging code; 
 - provides functions such as resampleing, dividing patches, restoring patches, etc.





# Environmental Preparation
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

## Demo1: Load original image and mask, voxel resampling, adjust window width and window level, 3D visualization


```python

from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # build instance
subject1.appendImageFromNifti('CT', img_path)  # Load image, custom modal name, such as 'CT'
subject1.appendSementicMaskFromNifti('CT', mask_path)  # Load the mask, pay attention to the corresponding modal name
# also can use appendImageAndSementicMaskFromNifti to load both image and mask at the same time

subject1.resample('CT', aim_spacing=[1, 1, 1])  # Resample to 1x1x1 mm (note the unit is mm)
subject1.adjst_Window('CT', WW = 321, WL = 123) # Adjust window width and window level

# 3D visualization
subject1.show_scan('CT', show_type='slice')  # Display original image in slice mode
subject1.show_scan('CT', show_type='volume')  # Display original image in volume mode
subject1.show_MaskAndScan('CT', show_type='volume') # Display original image and mask at the same time
subject1.show_bbox('CT', 2)  # Display the bbox shape. Note that when there is no bbox, the minimum external matrix is automatically generated from the mask as bbox

```


<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scan_slicemode.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scan_volumemode.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>Display original image in slice mode</th>
<th>Display original image in volume mode</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_scanandmask_volumemode.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/1_show_bbox_volumemode.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>Display original image and mask at the same time</th>
<th>show bbox shape</th>
</tr>

</table>


## Demo 2.Split or reorganize patches in any dimension

To be precise, it is to restore the patch to the corresponding position in the original space. If a patch passes through the segmentation network and outputs the segmentation result of the patch, one can be restored to the original position for visualization.


```python
from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama()  # build instance
subject1.appendImageFromNifti('CT', img_path)  # Load image, custom modal name, such as 'CT'
subject1.appendSementicMaskFromNifti('CT', mask_path)  # Load the mask, pay attention to the corresponding modal name
# also can use appendImageAndSementicMaskFromNifti to load both image and mask at the same time

subject1.resample('CT', aim_spacing=[1, 1, 1])  # Resample to 1x1x1 mm (note the unit is mm)
subject1.adjst_Window('CT', WW=321, WL=123)  # Adjust window width and window level

# smooth denoising
qwe = subject1.slice_neibor_add('CT', axis=['z'], add_num=[7], add_weights='Gaussian')  # Use a Gaussian kernel, smooth on the z-axis

# Extract the image in the bbox (bbox is the minimum external matrix of the segmentation mask)
bbox_image = subject1.getImagefromBbox('CT', ex_mode='square', ex_mms=24, aim_shape=[256, 256, 256])

"""
The logic of splitting patch:
     1) First frame the ROI to obtain the bbox, and then operate within the ROI
     2) External expansion roi
     3) Take out the image in the roi and scale it to aim_shape
     4) split patch
"""

# Split patch(setting 1): divide the patch along the Z axis, the patch is 2D, and take one layer every 10 layers
subject1.makePatch(mode='slideWinND',  # default is ok
				   img_type='CT',  # modality keyword
				   aim_shape=[256, 256, 256],  # scale to this size
				   slices=[256 // 2, 256 // 2, 1],  # The length of each patch in each dimension (note that the Z axis is 1, that is, 2D patches are divided along the Z axis)
				   stride=[256 // 2, 256 // 2, 10],  # The sliding window size of the patch in each axis (note that the z axis here is 10)
				   expand_r=[1, 1, 1],  # Similar to the expansion coefficient of dilated convolution (hole convolution), 1 means no expansion
				   ex_mode='square',  # After taking the bbox, turn the bbox into a cube
				   ex_mms=24,  # How many mm does the bbox expand (or after it becomes a cube)
				   )
reconstuct_img = slide_window_n_axis_reconstruct(subject1.patches['CT'])  # Put all the patches back into the original space
reconstuct_img_half = slide_window_n_axis_reconstruct(
	subject1.patches['CT'][:len(subject1.patches['CT']) // 2])  # Put half of the patches back into the original space

patch = subject1.patches['CT']  # get patches
show3D(np.concatenate([bbox_image, reconstuct_img], axis=1))
show3D(np.concatenate([bbox_image, reconstuct_img_half], axis=1))



# Split patch(setting 2)：Block (similar to Rubik's Cube)
subject1.makePatch(mode='slideWinND',  # default is ok
				   img_type='CT',  # modality keyword
				   aim_shape=[256, 256, 256],  # scale to this size
				   slices=[256 // 8, 256 // 8,  256 // 8],
				   stride=[( 256 // 8)+3, ( 256 // 8)+3, ( 256 // 8)+3],
				   expand_r=[1, 1, 1], 
				   ex_mode='square', 
				   ex_mms=24,
				   )
reconstuct_img = slide_window_n_axis_reconstruct(subject1.patches['CT'])
reconstuct_img_half = slide_window_n_axis_reconstruct(
	subject1.patches['CT'][:len(subject1.patches['CT']) // 2])

patch = subject1.patches['CT']  # 获取patch
show3D(np.concatenate([bbox_image, reconstuct_img], axis=1))
show3D(np.concatenate([bbox_image, reconstuct_img_half], axis=1))



# Split patch(setting 3)：Observe the influence of the expansion coefficient (in fact, it is basically useless)
subject1.makePatch(mode='slideWinND',
				   img_type='CT', 
				   aim_shape=[256, 256, 256], 
				   slices=[30, 30, 30], 
				   stride=[1, 1, 1], 
				   expand_r=[5, 5, 5],  # Similar to the expansion coefficient of dilated convolution (hole convolution), 1 means no expansion
				   ex_mode='square', 
				   ex_mms=24,  
				   )

reconstuct_img_onlyone = slide_window_n_axis_reconstruct([subject1.patches['CT'][0]])  # Put only one patch back into the original space (observe the effect of the expansion coefficient)

patch = subject1.patches['CT'] # get patches
show3D(np.concatenate([bbox_image, reconstuct_img_onlyone], axis=1))




```


<table>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_all_z.gif" height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_half_z.gif" height="283" width="324" alt="input heatmaps"></td>
</tr>

<tr>
<th>Setting 1: Split the patch along the Z axis and put back allpatch</th>
<th>Setting 1: Split the patch along the Z axis and put back half of the patch</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/2_show_patches_squared.gif"  height="283" width="324" alt="input images"></td>
<td><img src="https://github.com/WAMAWAMA/wama_medic/blob/master/pic/3_show_patches_expand.gif"  height="283" width="324" alt="input images"></td>
</tr>

<tr>
<th>Setting 2: Block (similar to Rubik's Cube)</th>
<th>Setting 3: Observe the effect of the expansion coefficient</th>
</tr>

</table>


## Demo 3.Image enhancement or augmentation (3D)

```python

from wama.utils import *
from wama.data_augmentation import aug3D

img_path = r'D:\git\testnini\s22_v1.nii.gz'
mask_path = r'D:\git\testnini\s22_v1_m1.nii.gz'

subject1 = wama()
subject1.appendImageAndSementicMaskFromNifti('CT', img_path, mask_path)
subject1.adjst_Window('CT', 321, 123)
bbox_image = subject1.getImagefromBbox('CT',ex_mode='square', aim_shape=[128,128,128])


bbox_image_batch = np.expand_dims(np.stack([bbox_image,bbox_image,bbox_image,bbox_image,bbox_image]),axis=1)# build batch
bbox_mask_batch = np.zeros(bbox_image_batch.shape)
bbox_mask_batch[:,:,20:100,20:100,20:100] = 1

auger = aug3D(size=[128,128,128], deformation_scale = 0.25) # The size can be the original image size (or batch size)
aug_result = auger.aug(dict(data=bbox_image_batch, seg = bbox_mask_batch))  # Note that it needs to be passed in as a dictionary

# visualization
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
<th>Original image, before and after amplification</th>
<th>mask, before and after amplification</th>
</tr>


</table>


## Demo 4.image cropping
```python
from wama.utils import *

img_path = r"D:\git\testnini\s1_v1.nii"
mask_path = r"D:\git\testnini\s1_v1_m1_w.nii"

subject1 = wama() # build instance
subject1.appendImageFromNifti('CT', img_path)  # Load image, custom modal name, such as 'CT'
subject1.appendSementicMaskFromNifti('CT', mask_path)  # Load the mask, pay attention to the corresponding modal name
# It is also possible to use appendImageAndSementicMaskFromNifti to load both image and mask at the same time

print(subject1.scan['CT'].shape)


# crop
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
