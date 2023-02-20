import os
# os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
# from multiprocessing import set_start_method
# set_start_method("spawn")
import json
import helper.misc as hmisc
import helper.array_transf as harray
import small_project.homogeneity_measure.metric_implementations as homog_metric
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_class as hplotc
import skimage.feature
import re
import objective.inhomog_removal.CalculateMetrics as CalcMetrics
import time
import scipy.stats
import helper.metric as hmetric
base_pred = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

"""
Define all paths for volunteer data..
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/volunteer_data/body_mask'
sel_mask_dir = body_mask_dir


single_biasf_volunteer = {"dinput": os.path.join(base_pred, 'single_biasfield/volunteer_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/volunteer_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "volunteer"}

multi_biasf_volunteer = {"dinput": os.path.join(base_pred, 'multi_biasfield/volunteer_corrected/input'),
                         "dpred": os.path.join(base_pred, 'multi_biasfield/volunteer_corrected/pred'),
                         "dmask": sel_mask_dir,
                          "name": "volunteer"}

multi_homog_volunteer = {"dinput": os.path.join(base_pred, 'multi_homogeneous/volunteer_corrected/input'),
                        "dpred": os.path.join(base_pred, 'multi_homogeneous/volunteer_corrected/pred'),
                        "dmask": sel_mask_dir,
                          "name": "volunteer"}

single_homog_volunteer = {"dinput": os.path.join(base_pred, 'single_homogeneous/volunteer_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/volunteer_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "volunteer"}

n4itk_volunteer = {"dinput": "/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk/input",
                   "dpred": "/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk/pred",
                   "dmask": sel_mask_dir,
                   "name": "volunteer"}



"""
Loool do this stuff for ... patient data
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/daan_reesink/mask'
sel_mask_dir = body_mask_dir


single_biasf_patient = {"dinput": os.path.join(base_pred, 'single_biasfield/patient_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/patient_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient"}

single_homog_patient = {"dinput": os.path.join(base_pred, 'single_homogeneous/patient_corrected/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/patient_corrected/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient"}

n4itk_patient = {"dinput": "/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk/input",
                    "dpred": "/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk/pred",
                    "dmask": sel_mask_dir,
                          "name": "patient"}

"""
Patient 3T paths 
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/mask'
sel_mask_dir = body_mask_dir


single_biasf_patient_3T = {"dinput": os.path.join(base_pred, 'single_biasfield/patient_corrected_3T/input'),
                          "dpred": os.path.join(base_pred, 'single_biasfield/patient_corrected_3T/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient_3T"}

single_homog_patient_3T = {"dinput": os.path.join(base_pred, 'single_homogeneous/patient_corrected_3T/input'),
                          "dpred": os.path.join(base_pred, 'single_homogeneous/patient_corrected_3T/pred'),
                          "dmask": sel_mask_dir,
                          "name": "patient_3T"}

n4itk_patient_3T = {"dinput": "/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/target",
                 "dpred": "/local_scratch/sharreve/mri_data/prostate_weighting_h5/test/target_corrected_N4",
                 "dmask": sel_mask_dir,
                          "name": "patient_3T"}


"""
Patient 1.5T paths 
"""

body_mask_dir = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask_b1'
fat_mask_dir = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/mask'
input_n4_test = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/input_abs_sum'
sel_mask_dir = body_mask_dir


single_biasf_test = {"dinput": input_n4_test, #os.path.join(base_pred, 'single_biasfield/target_corrected/input'),
                     "dpred": os.path.join(base_pred, 'single_biasfield/target_corrected/pred'),
                     "dtarget": os.path.join(base_pred, 'single_biasfield/target_corrected/target'),
                     "dmask": sel_mask_dir,
                     "dmask_fat": fat_mask_dir,
                     "name": "test"}

single_homog_test = {"dinput": input_n4_test, #os.path.join(base_pred, 'single_homogeneous/target_corrected/input'),
                     "dpred": os.path.join(base_pred, 'single_homogeneous/target_corrected/pred'),
                     "dtarget": os.path.join(base_pred, 'single_homogeneous/target_corrected/target'),
                     "dmask": sel_mask_dir,
                     "dmask_fat": fat_mask_dir,
                     "name": "test"}

# Using single as input.. to avoid any differences...
multi_biasf_test = {"dinput": os.path.join(base_pred, 'single_biasfield/target_corrected/input'),
                    "dpred": os.path.join(base_pred, 'multi_biasfield/target_corrected/pred'),
                    "dtarget": os.path.join(base_pred, 'multi_biasfield/target_corrected/target'),
                    "dmask": sel_mask_dir,
                    "dmask_fat": fat_mask_dir,
                    "name": "test"}

# Using single as input.. to avoid any differences...
multi_homog_test = {"dinput": os.path.join(base_pred, 'single_homogeneous/target_corrected/input'),
                    "dpred": os.path.join(base_pred, 'multi_homogeneous/target_corrected/pred'),
                    "dtarget": os.path.join(base_pred, 'multi_homogeneous/target_corrected/target'),
                    "dmask": sel_mask_dir,
                    "dmask_fat": fat_mask_dir,
                    "name": "test"}

n4itk_test = {"dinput": input_n4_test,
              "dpred": "/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/corrected_N4",
              "dtarget": "/local_scratch/sharreve/mri_data/registrated_h5/test_nifti/target",
              "dmask": sel_mask_dir,
              "dmask_fat": fat_mask_dir,
              "name": "test"}

# Volunteer set
volunteer_list = [single_biasf_volunteer, single_homog_volunteer, multi_biasf_volunteer, multi_homog_volunteer, n4itk_volunteer]
# Patient 3T set
patient_3T_list = [single_biasf_patient_3T, single_homog_patient_3T, n4itk_patient_3T]
# Patient 7T set
patient_7T_list = [single_biasf_patient, single_homog_patient, n4itk_patient]
# Test set
test_list = [single_biasf_test, multi_biasf_test, single_homog_test, multi_homog_test, n4itk_test]
# test_list = [n4itk_test]
# test_list = [single_homog_test, multi_homog_test]
# test_list = [multi_biasf_test]
#  test_list + volunteer_list + patient_7T_list + patient_3T_list:
calculate_no_target_metric = True
for i_dict in volunteer_list + patient_3T_list + patient_7T_list + test_list:
    # Options for testing stuff...
    i_dict['mid_slice'] = True
    i_dict['debug'] = False
    i_dict['shrink_pixels'] = 30
    #
    dataset_name = i_dict['name']
    print(f"Input path {i_dict['dinput']}")
    mask_dir_name = os.path.basename(i_dict['dmask'])
    model_name = os.path.basename(os.path.dirname(os.path.dirname(i_dict['dinput'])))
    mask_name = re.sub('_mask', '', mask_dir_name)
    dest_dir = os.path.dirname(i_dict['dpred'])
    if dataset_name == 'volunteer':
        # Needed for volunteer 7T data
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=10*10, **i_dict)
        metric_obj.glcm_dist = list(range(10))[1:]
    elif dataset_name == 'patient_3T':
        # Needed for patient 3T
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.h5', patch_size=7*10, mask_suffix='_target', **i_dict)
        metric_obj.glcm_dist = list(range(7))[1:]
    elif dataset_name == 'patient':
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=16*10, **i_dict)
        # We want 5mm and we have a pixel spacing of approx 0.28mm
        metric_obj.glcm_dist = list(range(16))[1:]
    elif dataset_name == 'test':
        # In the test set we have a pixel spacing of...
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=7*10, **i_dict)
        # We want 5mm and we have a pixel spacing of approx 0.7mm
        metric_obj.glcm_dist = list(range(7))[1:]
    # print(" ONLY USING THREE FILES TO MAKE SURE THAT WE CAN QUICKLY CHECK SOME RESULTS")
    # metric_obj.file_list = metric_obj.file_list[0:3]
    if calculate_no_target_metric:
        glcm_rel, glcm_input, glcm_pred, coef_var_rel, coef_var_input, coef_var_pred, slice_list = metric_obj.run_features()
        glcm_relative_change_dict = hmisc.listdict2dictlist(glcm_rel)
        # Super ugly.. /care
        glcm_relative_change_dict['file_list'] = metric_obj.file_list
        glcm_relative_change_dict['slice_list'] = slice_list
        glcm_input_dict = hmisc.listdict2dictlist(glcm_input)
        glcm_input_dict['file_list'] = metric_obj.file_list
        glcm_input_dict['slice_list'] = slice_list
        glcm_pred_dict = hmisc.listdict2dictlist(glcm_pred)
        glcm_pred_dict['file_list'] = metric_obj.file_list
        glcm_pred_dict['slice_list'] = slice_list
        np.save(os.path.join(dest_dir, f'{mask_name}_rel_coef_of_variation.npy'), coef_var_rel)
        np.save(os.path.join(dest_dir, f'{mask_name}_input_coef_of_variation.npy'), coef_var_input)
        np.save(os.path.join(dest_dir, f'{mask_name}_pred_coef_of_variation.npy'), coef_var_pred)
        relative_change_ser = json.dumps(glcm_relative_change_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_rel_change_glcm.json'), 'w') as f:
           f.write(relative_change_ser)
        input_ser = json.dumps(glcm_input_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_input_change_glcm.json'), 'w') as f:
           f.write(input_ser)
        pred_ser = json.dumps(glcm_pred_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_pred_change_glcm.json'), 'w') as f:
           f.write(pred_ser)

    if metric_obj.dtarget is not None:
        coefv_target_rel, coefv_target, glcm_rel, glcm_target, RMSE_list, SSIM_list, WSS_distance, slice_list = metric_obj.run_features_target()
        glcm_target_change_dict = hmisc.listdict2dictlist(glcm_target)
        glcm_target_change_dict['file_list'] = metric_obj.file_list
        glcm_target_change_dict['slice_list'] = slice_list
        glcm_target_relative_change_dict = hmisc.listdict2dictlist(glcm_rel)
        glcm_target_relative_change_dict['file_list'] = metric_obj.file_list
        glcm_target_relative_change_dict['slice_list'] = slice_list
        target_change_ser = json.dumps(glcm_target_change_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_target_change_glcm.json'), 'w') as f:
            f.write(target_change_ser)
        rel_target_change_ser = json.dumps(glcm_target_relative_change_dict)
        with open(os.path.join(dest_dir, f'{mask_name}_rel_target_change_glcm.json'), 'w') as f:
            f.write(rel_target_change_ser)

        np.save(os.path.join(dest_dir, f'{mask_name}_rel_target_coef_of_variation.npy'), coefv_target_rel)
        np.save(os.path.join(dest_dir, f'{mask_name}_target_coef_of_variation.npy'), coefv_target)
        np.save(os.path.join(dest_dir, f'{mask_name}_rmse_values.npy'), RMSE_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_ssim_values.npy'), SSIM_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_wasserstein_values.npy'), WSS_distance)


# Nice read..? For later...
# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0212110&type=printable
# Gray-level invariant Haralick texture features
