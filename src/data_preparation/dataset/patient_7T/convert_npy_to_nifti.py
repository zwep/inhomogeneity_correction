
import numpy as np
import os
import nibabel

dorig = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image'
ddest = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/Daan_filter/image_nifti'

for sel_file in os.listdir(dorig):
    file_name, ext = os.path.splitext(sel_file)
    file_path = os.path.join(dorig, sel_file)
    dest_file_path = os.path.join(ddest, file_name + '.nii.gz')
    temp_A = np.load(file_path)
    temp_A = temp_A.T[::-1, ::-1]
    nibabel_obj = nibabel.Nifti1Image(temp_A, np.eye(4))
    nibabel.save(nibabel_obj, dest_file_path)