# 复发文章的分割图制作，动脉期和静脉期
# 输入的图片应该是经过归一化的
import torch
from segmentation_models_pytorch_w import Unet
from wama.utils import *



id = 5
unet_path = r"E:\ubuntu\zhuomian\mianyizuhua_data\@pretrained\Unet_ef6_attention_mode1_256\models\best_unet_score.pkl"
unet_path2 = r"E:\ubuntu\zhuomian\mianyizuhua_data\@pretrained\Unet_ef6_attention_mode1_256\models\epoch11_Testdice0.7802.pkl"
nii_path = r"E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\aWITHmask4radiomics\s"+str(id)+"_v1.nii"
mask_path = r"E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\aWITHmask4radiomics\s"+str(id)+"_v1_m1_w.nii"
nii_path_v = r"E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\vWITHmask4radiomics\s"+str(id)+"_v1.nii"
mask_path_v = r"E:\@data_NENs\@data_NENs_recurrence\or_data\data\nii\vWITHmask4radiomics\s"+str(id)+"_v1_m1.nii"


object1 = wama()
object1.appendImageAndSementicMaskFromNifti('CT_a', nii_path, mask_path)
object1.appendImageAndSementicMaskFromNifti('CT_v', nii_path_v, mask_path_v)
object1.appendImageAndSementicMaskFromNifti('CT_v4show', nii_path_v, mask_path_v)
object1.adjst_Window('CT_a',WW = 310 ,WL = 130) # 动脉期
object1.adjst_Window('CT_v',WW = 320 ,WL = 0) # 静脉期
object1.adjst_Window('CT_v4show',WW = 320 ,WL = 120) # 静脉期

scan_a = object1.getImagefromBbox('CT_a', ex_voxels=[20,20,20], ex_mode='square', aim_shape=[256,256,None])
mask_a = object1.getMaskfromBbox('CT_a', ex_voxels=[20,20,20], ex_mode='square', aim_shape=[256,256,None])
scan_scale_a = mat2gray(scan_a)

scan_v = object1.getImagefromBbox('CT_v', ex_voxels=[20,20,20], ex_mode='square', aim_shape=[256,256,None])
mask_v = object1.getMaskfromBbox('CT_v', ex_voxels=[20,20,20], ex_mode='square', aim_shape=[256,256,None])
scan_scale_v = mat2gray(scan_v)

scan_v4show = object1.getImagefromBbox('CT_v4show', ex_voxels=[20,20,20], ex_mode='square', aim_shape=[256,256,None])
scan_scale_v4show = mat2gray(scan_v4show)

model = Unet(encoder_name="efficientnet-b6",
                     encoder_weights=None,
                     in_channels=1, classes=1, encoder_depth=5, decoder_attention_type='scse')
model.load_state_dict(torch.load(unet_path,map_location=torch.device('cpu')))



scan_tensor_a = torch.tensor(scan_scale_a[:,:,scan_scale_a.shape[2]//2])
scan_tensor_a = torch.unsqueeze(scan_tensor_a,0)
scan_tensor_a = torch.unsqueeze(scan_tensor_a,0)
predict_mask = model(scan_tensor_a)[0]
predict_mask = torch.sigmoid(predict_mask)
predict_mask = torch.squeeze(predict_mask)
predict_mask_a = predict_mask.data.cpu().numpy()

model.load_state_dict(torch.load(unet_path2,map_location=torch.device('cpu')))
scan_tensor_v = torch.tensor(scan_scale_v[:,:,scan_scale_v.shape[2]//2])
scan_tensor_v = torch.unsqueeze(scan_tensor_v,0)
scan_tensor_v = torch.unsqueeze(scan_tensor_v,0)
predict_mask = model(scan_tensor_a)[0]
predict_mask = torch.sigmoid(predict_mask)
predict_mask = torch.squeeze(predict_mask)
predict_mask_v = predict_mask.data.cpu().numpy()



plt.subplot(2,3,1)
plt.imshow(scan_scale_a[:,:,scan_scale_a.shape[2]//2],cmap=plt.cm.gray)
plt.subplot(2,3,2)
plt.imshow(mask_a[:,:,mask_a.shape[2]//2],cmap=plt.cm.gray)
plt.subplot(2,3,3)
plt.imshow(predict_mask_a,cmap=plt.cm.gray)
plt.subplot(2,3,4)
plt.imshow(scan_scale_v4show[:,:,scan_scale_v4show.shape[2]//2],cmap=plt.cm.gray)
plt.subplot(2,3,5)
plt.imshow(mask_v[:,:,mask_v.shape[2]//2],cmap=plt.cm.gray)
plt.subplot(2,3,6)
plt.imshow(predict_mask_v,cmap=plt.cm.gray)
plt.show()

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).astype(np.float)
    Inter = np.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + 1e-6)
    return DC

# show2D(scan_scale_v4show[:,:,scan_scale_v4show.shape[2]//2])
# show2D(scan_scale_v[:,:,scan_scale_v4show.shape[2]//2])

print('a:',get_DC(predict_mask_a, mask_a[:,:,mask_a.shape[2]//2]))
print('v:',get_DC(predict_mask_v, mask_v[:,:,mask_v.shape[2]//2]))



















