"""
I dont trust the way the data is created...
Here we create some code that can be run remotely to store some input examples...
"""
from skimage import img_as_ubyte
import data_generator.InhomogRemoval as data_gen
import helper.plot_class as hplotc
import numpy as np
import helper.array_transf as harray
import helper.misc as hmisc
import nibabel
import os

dir_data = '/local_scratch/sharreve/mri_data/registrated_h5'
ddest = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti'
ddest_input = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input'
ddest_input_abs_sum = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input_abs_sum'
ddest_target = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target'
ddest_target_biasf = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target_biasf'
ddest_mask = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/mask'

gen = data_gen.DataGeneratorInhomogRemovalH5(ddata=dir_data,
                                             dataset_type='test', complex_type='cartesian',
                                             file_ext='h5',
                                             masked=True,
                                             debug=False,
                                             relative_phase=True,
                                             transform_type='complex',
                                             transform_type_target='real',
                                             target_type='both',
                                             shuffle=False,
                                             cycle_all=True,
                                             SNR_mode=10)

prev_file = ''
temp_input = []
temp_mask = []
temp_input_abs = []
temp_target = []
temp_target_biasf = []
# gen.container_file_info[0]['slice_count']
for i_file in np.arange(gen.__len__()):
    cur_file = gen.container_file_info[0]['file_list'][i_file]
    print(cur_file, end='\r')
    if (cur_file != prev_file) and len(temp_input) > 0:
        file_name, _ = os.path.splitext(prev_file)
        input_stacked = np.stack(temp_input, axis=1)
        # Used to save some memory.. bleh
        input_stacked = harray.scale_minmax(input_stacked, is_complex=True, axis=(0, -2, -1))
        input_stacked_split = np.stack([input_stacked.real, input_stacked.imag], axis=0)
        test = img_as_ubyte(input_stacked_split)
        print('Shape of input stacked ', test.shape)
        # Using NPY because that is easier with my data...
        np.save(os.path.join(ddest_input, f"{file_name}.npy"), test)
        # Store input
        input_abs_stacked = np.concatenate(temp_input_abs)
        print('Shape of abs input stacked ', input_abs_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(input_abs_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_input_abs_sum, f"{file_name}.nii.gz"))
        # Store target
        target_stacked = np.concatenate(temp_target)
        print('Shape of target ', target_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(target_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_target, f"{file_name}.nii.gz"))
        # Store mask
        mask_stacked = np.concatenate(temp_mask)
        print('Shape of mask ', mask_stacked.shape)
        nibabel_obj = nibabel.Nifti1Image(mask_stacked.T[::-1, ::-1], np.eye(4))
        nibabel.save(nibabel_obj, os.path.join(ddest_mask, f"{file_name}.nii.gz"))
        temp_input = []
        temp_mask = []
        temp_input_abs = []
        temp_target = []
    cont = gen.__getitem__(i_file)
    input_array = np.abs(cont['input'].numpy()[::2] + 1j * cont['input'].numpy()[1::2])
    input_abs = np.sqrt(cont['input'].numpy()[::2] ** 2 + cont['input'].numpy()[1::2] ** 2).sum(axis=0)
    input_abs = harray.scale_minmax(input_abs)
    input_abs = img_as_ubyte(input_abs)[None]
    target = cont['target'].numpy()
    target = harray.scale_minmax(target)
    target = img_as_ubyte(target)
    mask = cont['mask'].numpy()
#
# inp_masked = np.ma.masked_where(1-mask[0:1], input_abs)
# fig_obj = hplotc.ListPlot([inp_masked], ax_off=True)
# fig_obj.figure.savefig('/local_scratch/sharreve/input_test.png', bbox_inches='tight', pad_inches=0.0)
# fig_obj = hplotc.ListPlot([target[0]], ax_off=True)
# fig_obj.figure.savefig('/local_scratch/sharreve/input_test.png', bbox_inches='tight', pad_inches=0.0)
    # Store it in a temp array..
    temp_input.append(input_array)
    temp_mask.append(mask)
    temp_input_abs.append(input_abs)
    temp_target.append(target)
    prev_file = cur_file
