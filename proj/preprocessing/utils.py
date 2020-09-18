# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)
# Source: https://git.ee.ethz.ch/baumgach/discriminative_learning_toolbox/blob/master/utils.py

import nibabel as nib
import numpy as np
import os
import logging
from skimage import measure, transform
import SimpleITK as sitk
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:

    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        out = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp, borderMode=cv2.BORDER_REPLICATE)
        return np.reshape(out, img.shape)

    def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):
        onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
        return np.argmax(onehot_output, axis=-1)

    def resize_image(im, size, interp=cv2.INTER_LINEAR):
        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        #add last dimension again if it was removed by resize
        if im.ndim > im_resized.ndim:
            im_resized = np.expand_dims(im_resized, im.ndim)
        return im_resized

    def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
        return np.argmax(onehot_output, axis=-1)

    def deformation_to_transformation(dx, dy):

        nx, ny = dx.shape

        grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        return map_x, map_y

    def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR):

        map_x, map_y = deformation_to_transformation(dx, dy)

        do_optimization = (interp == cv2.INTER_LINEAR)
        # The following command converts the maps to compact fixed point representation
        # this leads to a ~20% increase in speed but could lead to accuracy losses
        # Can be uncommented
        if do_optimization:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)

        remapped = cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))
        if im.ndim > remapped.ndim:
            remapped = np.expand_dims(remapped, im.ndim)
        return remapped


    def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR, do_optimisation=True):

        onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp, do_optimisation=do_optimisation)
        return np.argmax(onehot_output, axis=-1)


def convert_to_onehot(lblmap, nlabels):

    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:,:,ii] = (lblmap == ii).astype(np.uint8)
    return output

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a,v)


def norm_l2(a,v):

    a = a.flatten()
    v = v.flatten()

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)

    return np.mean(np.sqrt(a**2 + v**2))



def all_argmax(arr, axis=None):

    return np.argwhere(arr == np.amax(arr, axis=axis))


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

# read nii image from dir path
def read_nii_image(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# save matrix to nii file
def saveArray2nii(image,path):
    image_nii = sitk.GetImageFromArray(image)
    sitk.WriteImage(image_nii, path)

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def create_and_save_nii(data, img_path):

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, img_path)



class Bunch:
    # Useful shortcut for making struct like contructs
    # Example:
    # mystruct = Bunch(a=1, b=2)
    # print(mystruct.a)
    # >>> 1
    def __init__(self, **kwds):
        self.__dict__.update(kwds)



def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):

    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.

    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o

    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o

    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)

    image = (np.divide((image - min_i), max_i - min_i) * (max_o - min_o) + min_o).copy()

    image[image > max_o] = max_o
    image[image < min_o] = min_o

    return image


def map_images_to_intensity_range(X, min_o, max_o, percentiles=0):

    X_mapped = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_mapped[ii,...] = map_image_to_intensity_range(Xc, min_o, max_o, percentiles)

    return X_mapped.astype(np.float32)


def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img
