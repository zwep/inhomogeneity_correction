"""
Remote evaluation... for the test set..

also the data paths should be set to the test split
"""

import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import helper.plot_class as hplotc
import os

# needed these for some misc coding
import helper.array_transf as harray
import matplotlib.pyplot as plt
import numpy as np
import helper.misc as hmisc


input_dir_volunteer = '/home/sharreve/local_scratch/mri_data/volunteer_data/t2w'
mask_dir_volunteer = '/home/sharreve/local_scratch/mri_data/volunteer_data/body_mask'

input_dir_patient = '/home/sharreve/local_scratch/mri_data/daan_reesink/image'
mask_dir_patient = '/home/sharreve/local_scratch/mri_data/daan_reesink/mask'

input_dir_patient_3T = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/target'
mask_dir_patient_3T = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/mask'

input_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/input'
input_dir_test_som = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/input_abs_sum'
# mask_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/mask'
mask_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/mask_b1'
target_dir_test = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/target'

base_model_dir = '/home/sharreve/local_scratch/model_run/selected_inhomog_removal_models'
biasf_single_config_path = os.path.join(base_model_dir, 'single_biasfield')
homog_single_config_path = os.path.join(base_model_dir, 'single_homogeneous')
both_single_config_path = os.path.join(base_model_dir, 'single_both')
both_multi_config_path = os.path.join(base_model_dir, 'multi_both')
biasf_multi_config_path = os.path.join(base_model_dir, 'multi_biasfield')
homog_multi_config_path = os.path.join(base_model_dir, 'multi_homogeneous')

"""
Data paths for test split
"""

# These are the dicts for patient evaluation.
biasf_single_dict_test = {"dconfig": biasf_single_config_path, "dimage": input_dir_test_som, "dmask": mask_dir_test,
                          "ddest": os.path.join(biasf_single_config_path, "target_corrected"),
                          "dtarget": target_dir_test}

homog_single_dict_test = {"dconfig": homog_single_config_path, "dimage": input_dir_test_som, "dmask": mask_dir_test,
                          "ddest": os.path.join(homog_single_config_path, "target_corrected"),
                          "dtarget": target_dir_test}

biasf_multi_dict_test = {"dconfig": biasf_multi_config_path, "dimage": input_dir_test, "dmask": mask_dir_test,
                         "ddest": os.path.join(biasf_multi_config_path, "target_corrected"),
                        "dtarget": target_dir_test}

homog_multi_dict_test = {"dconfig": homog_multi_config_path, "dimage": input_dir_test, "dmask": mask_dir_test,
                         "ddest": os.path.join(homog_multi_config_path, "target_corrected"),
                         "dtarget": target_dir_test}

both_single_dict_test = {"dconfig": both_single_config_path, "dimage": input_dir_test_som, "dmask": mask_dir_test,
                         "ddest": os.path.join(both_single_config_path, "target_corrected")}

both_mult_dict_test = {"dconfig": both_multi_config_path, "dimage": input_dir_test, "dmask": mask_dir_test,
                       "ddest": os.path.join(both_multi_config_path, "target_corrected")}

"""
Stuff for 3T data
"""

# These are the dicts for patient evaluation.
biasf_single_dict_patient_3T = {"dconfig": biasf_single_config_path, "dimage": input_dir_patient_3T, "dmask": mask_dir_patient_3T,
                                "ddest": os.path.join(biasf_single_config_path, "patient_corrected_3T")}
homog_single_dict_patient_3T = {"dconfig": homog_single_config_path, "dimage": input_dir_patient_3T, "dmask": mask_dir_patient_3T,
                                "ddest": os.path.join(homog_single_config_path, "patient_corrected_3T")}

both_single_dict_patient_3T = {"dconfig": both_single_config_path, "dimage": input_dir_patient_3T, "dmask": mask_dir_patient_3T,
                                "ddest": os.path.join(both_single_config_path, "patient_corrected_3T")}

"""
Stuff for patient data..
"""

# These are the dicts for patient evaluation.
biasf_single_dict_patient = {"dconfig": biasf_single_config_path, "dimage": input_dir_patient, "dmask": mask_dir_patient,
                             "ddest": os.path.join(biasf_single_config_path, "patient_corrected")}

homog_single_dict_patient = {"dconfig": homog_single_config_path, "dimage": input_dir_patient, "dmask": mask_dir_patient,
                             "ddest": os.path.join(homog_single_config_path, "patient_corrected")}

both_single_dict_patient = {"dconfig": both_single_config_path, "dimage": input_dir_patient, "dmask": mask_dir_patient,
                            "ddest": os.path.join(both_single_config_path, "patient_corrected")}

"""
Stuff for volunteer data...
"""

# These are the dicts for patient evaluation.
biasf_single_dict_volunteer = {"dconfig": biasf_single_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                     "ddest": os.path.join(biasf_single_config_path, "volunteer_corrected")}
homog_single_dict_volunteer = {"dconfig": homog_single_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                     "ddest": os.path.join(homog_single_config_path, "volunteer_corrected")}
biasf_multi_dict_volunteer = {"dconfig": biasf_multi_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                     "ddest": os.path.join(biasf_multi_config_path, "volunteer_corrected")}
homog_multi_dict_volunteer = {"dconfig": homog_multi_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                     "ddest": os.path.join(homog_multi_config_path, "volunteer_corrected")}

both_single_dict_volunteer = {"dconfig": both_single_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                            "ddest": os.path.join(both_single_config_path, "volunteer_corrected")}
both_mult_dict_volunteer = {"dconfig": both_multi_config_path, "dimage": input_dir_volunteer, "dmask": mask_dir_volunteer,
                            "ddest": os.path.join(both_multi_config_path, "volunteer_corrected")}

dict_list_test = [biasf_single_dict_test, biasf_multi_dict_test, homog_single_dict_test, homog_multi_dict_test, ]
dict_list_vol = [biasf_single_dict_volunteer, biasf_multi_dict_volunteer, homog_multi_dict_volunteer, homog_single_dict_volunteer]
dict_list_pat_3T = [biasf_single_dict_patient_3T, homog_single_dict_patient_3T]
dict_list_pat = [biasf_single_dict_patient, homog_single_dict_patient]

dict_list_test_biasf = [biasf_single_dict_test, biasf_multi_dict_test]
dict_list_vol_biasf = [biasf_single_dict_volunteer, biasf_multi_dict_volunteer]
dict_list_pat_3T_biasf = [biasf_single_dict_patient_3T]
dict_list_pat_biasf = [biasf_single_dict_patient]
# Calculate everything for `both` models
both_model_list = [both_single_dict_volunteer, both_mult_dict_volunteer, both_single_dict_test, both_mult_dict_test, both_single_dict_patient_3T, both_single_dict_patient]

for temp_dict in [biasf_single_dict_patient]:
    base_name = os.path.basename(temp_dict['ddest'])
    model_name = os.path.basename(os.path.dirname(temp_dict['ddest']))
    hmisc.create_datagen_dir(temp_dict['ddest'], data_list=[], type_list=['input', 'biasfield', 'pred', 'mask', 'target'])
    mask_ext = '.nii.gz'
    mask_suffix = ''
    stride = 64
    if 'volunteer' in base_name:
        mask_ext = '.npy'
        mask_suffix = ''
        file_list = ["v9_03032021_1647583_11_3_t2wV4.nii.gz",
                     "v9_11022021_1643158_8_3_t2wV4.nii.gz",
                     "v9_10022021_1725072_12_3_t2wV4.nii.gz",
                     "v9_18012021_0939588_10_3_t2wV4.nii.gz"]
    elif base_name.endswith('3T'):
        mask_ext = '.h5'
        mask_suffix = '_target'
        file_list = ['8_MR.nii.gz',
                     '19_MR.nii.gz',
                     '41_MR.nii.gz',
                     '45_MR.nii.gz']
    elif base_name.endswith('patient_corrected'):
        mask_ext = '.npy'
        stride = 128
        file_list = ["7TMRI002.npy", "7TMRI005.npy", "7TMRI016.npy", "7TMRI020.npy"]
    else:
        print('File list is empty')
        file_list = []

    target_dir = temp_dict.get('dtarget', None)
    postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=temp_dict['dimage'],
                                                           mask_dir=temp_dict['dmask'],
                                                           dest_dir=temp_dict['ddest'],
                                                           target_dir=target_dir,
                                                           config_path=temp_dict['dconfig'],
                                                           executor_module=executor, config_name='config_param.json',
                                                           stride=stride, patch_shape=(256, 256),
                                                           storage_extension='nii',
                                                           mask_ext=mask_ext,
                                                           mask_suffix=mask_suffix)
    # Number 3 really produces nice images...
    # But is in theory IMO not the best... we are averaging anatomical images..
    # Number 2 averages the bias fields.. which should, in my idea, work better..
    # postproc_obj.experimental_postproc_both = 3
    # Run only specific files..
    postproc_obj.file_list = file_list
    postproc_obj.run()
    # import time
    # postproc_obj.load_file(0)
    # t0 = time.time()
    # postproc_obj.run_slice_patched(0)
    # print(time.time() - t0)
    # # postproc_obj.run_iterative_recon()

