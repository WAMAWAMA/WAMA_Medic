from skimage import io
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import numpy as np
def mat2gray(image):
	"""
	归一化函数（线性归一化）
	:param image: ndarray
	:return:
	"""
	# as dtype = np.float32
	image = image.astype(np.float32)
	image = (image - np.min(image)) / (np.max(image)-np.min(image)+ 1e-14)
	return image

def show2D(img2D):
	plt.imshow(img2D,cmap=plt.cm.gray)
	plt.show()

# 读取数据，分离通道
pic = io.imread(r'D:\1.tif')
pic_gamma = pic[:,:,1]
pic_hcr = pic[:,:,2]
show2D(pic_gamma)

# 检测gamma的高亮点
pic_gamma_mask = (mat2gray(pic_gamma)>=0.5).astype(np.float)
from skimage import measure, color
pic_gamma_mask_point = measure.label(pic_gamma_mask, connectivity=2)


# 计算每个高亮点的 hcr/gamma，以及对应点的面积
final_list = []
radius = 3
index_mask = np.zeros(pic_gamma_mask_point.shape) # 写上数字
index_mask = Image.fromarray(index_mask)
d = ImageDraw.Draw(index_mask)
fnt = ImageFont.truetype(r"D:\30780615671.ttf", 30)
for point_index in range(1, np.max(pic_gamma_mask_point)+1):
	# 记录坐标
	dim = np.where(pic_gamma_mask_point ==point_index)
	dim0, dim1 = [dim[0][0], dim[1][0]]

	# 读取point所在范围
	bbox = [np.min(dim[0]), np.max(dim[0]), np.min(dim[1]), np.max(dim[1])]
	bbox[0] -= radius
	bbox[1] += radius
	bbox[2] -= radius
	bbox[3] += radius

	# 计算强度比
	intens = (np.sum(pic_hcr[bbox[0]:bbox[1], bbox[2]:bbox[3]])) / (np.sum(pic_gamma[bbox[0]:bbox[1], bbox[2]:bbox[3]])+1e-12)

	# 记录
	final_list.append([point_index, intens])

	# 在mask写上数字
	d.text((dim1,dim0), str(point_index), font=fnt)

index_mask = np.array(index_mask)
show2D(index_mask)

# 在这里选点
index = [8,9,12,19] # 点的序号
for i in index:
	for ii in final_list:
		if ii[0] == i:
			print(i, ':',ii[1])









