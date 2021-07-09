from wama.utils import *
from wama.utils import *
import numpy as np
import copy
import os
sep = os.sep

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
            if file.endswith('.' + expname) and '_' not in file:
                file_List.append(os.path.join(filepath, file))
    return file_List


img_path = r"F:\9database\database\NIH-pancreas\Pancreas-CT\mask"
file_list = get_filelist_frompath(img_path, 'nii.gz')
save_path = r'F:\9database\database\NIH-pancreas\Pancreas-CT\mask_new'

for file in file_list:
    newpth = save_path+sep+file.split(sep)[-1]
    subject1 = wama()
    subject1.appendImageFromNifti('PET', file)
    subject1.scan['PET'] = subject1.scan['PET'][::-1,:,:]
    writeIMG(newpth,subject1.scan['PET'],subject1.spacing['PET'],subject1.origin['PET'],subject1.transfmat['PET'])
