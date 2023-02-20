import helper.misc as hmisc
import os
import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
from papers.inhomog_removal.helper_inhomog import RESULT_PATH

"""
This testing has become the real thing...
"""

fontsize = 16

# Remote destination directory
dest_dir_volunteer = '/home/sharreve/local_scratch/paper/inhomog/model_results/volunteer'
dest_dir_test_split = '/home/sharreve/local_scratch/paper/inhomog/model_results/test_split'
dest_dir_patient_7T = '/home/sharreve/local_scratch/paper/inhomog/model_results/patient_7T'
dest_dir_patient_3T = '/home/sharreve/local_scratch/paper/inhomog/model_results/patient_3T'

base_model_dir = '/home/sharreve/local_scratch/model_run/selected_inhomog_removal_models'
biasf_single_config_path = os.path.join(base_model_dir, 'single_biasfield')
homog_single_config_path = os.path.join(base_model_dir, 'single_homogeneous')
biasf_multi_config_path = os.path.join(base_model_dir, 'multi_biasfield')
homog_multi_config_path = os.path.join(base_model_dir, 'multi_homogeneous')

patient_7T_n4 = '/home/sharreve/local_scratch/mri_data/daan_reesink/image_n4itk/pred'
patient_3T_n4 = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5/test/target_corrected_N4'
volunteer_n4 = '/home/sharreve/local_scratch/mri_data/volunteer_data/t2w_n4itk/pred'
# 24 april...
test_n4 = '/home/sharreve/local_scratch/mri_data/registrated_h5/test_nifti/corrected_N4_new'

# Combine the possible interesting configs
pred_config_all = [biasf_single_config_path, homog_single_config_path, biasf_multi_config_path, homog_multi_config_path]
pred_config_single_channel = [biasf_single_config_path, homog_single_config_path]

dict_volunteer = {'config_list': pred_config_all,
                  'dataset': 'volunteer_corrected',
                  'ddest': dest_dir_volunteer,
                  'n4': volunteer_n4,
                  'ext': '.nii.gz',
                  'file_list': ["v9_03032021_1647583_11_3_t2wV4.nii.gz",
                                "v9_11022021_1643158_8_3_t2wV4.nii.gz",
                                "v9_10022021_1725072_12_3_t2wV4.nii.gz",
                                "v9_18012021_0939588_10_3_t2wV4.nii.gz"]}

dict_patient_7T = {'config_list': pred_config_single_channel,
                   'dataset': 'patient_corrected',
                   'ddest': dest_dir_patient_7T,
                   'n4': patient_7T_n4,
                   'ext': '.nii.gz',
                   'file_list': ["7TMRI002.nii.gz",
                                 "7TMRI005.nii.gz",
                                 "7TMRI016.nii.gz",
                                 "7TMRI020.nii.gz"]}

dict_patient_3T = {'config_list': pred_config_single_channel,
                   'dataset': 'patient_corrected_3T',
                   'ddest': dest_dir_patient_3T,
                   'n4': patient_3T_n4,
                   'ext': '.nii.gz',
                   'ext_n4': '.h5',
                   'file_list': ['8_MR.nii.gz',
                                 '19_MR.nii.gz',
                                 '41_MR.nii.gz',
                                 '45_MR.nii.gz']}

dict_test = {'config_list': pred_config_all,
             'dataset': 'target_corrected',
             'ddest': dest_dir_test_split,
             'n4': test_n4,
             'ext': '.nii.gz',
             'file_list': ["M20_to_51_MR_20200928_0003_transversal.nii.gz",
                           "M23_to_48_MR_20210127_0002_transversal.nii.gz",
                           "M20_to_8_MR_20210401_0003_transversal.nii.gz",
                           "M20_to_9_MR_20210324_0009_transversal.nii.gz"]}

# dict_test  - done
# dict_volunteer - done
for tempdict in [dict_test, dict_volunteer, dict_patient_3T, dict_patient_7T]:
    list_of_model_result = tempdict['config_list']
    dataset_midfix = tempdict['dataset']
    ddest = tempdict['ddest']
    dir_n4_results = tempdict['n4']
    file_ext_model_result = tempdict['ext']
    # file_list = tempdict['file_list']
    file_list = os.listdir(dir_n4_results)
    print('Using dataset ', dataset_midfix)
    print('Visualizating these files ', file_list)
    print('For these config-files ', list_of_model_result)
    for i_file in file_list:
        print(f'\t Processing file {i_file}')
        i_file_no_ext = hmisc.get_base_name(i_file)
        # Load the N4-corrected file
        ext_n4 = tempdict.get('ext_n4', tempdict['ext'])
        n4_file_path = os.path.join(dir_n4_results, i_file_no_ext + ext_n4)
        result_n4 = hmisc.load_array(n4_file_path)
        if ext_n4.endswith('.nii.gz'):
            result_n4 = result_n4.T[:, ::-1, ::-1]
        # Set the input, mask and target directories. These are the same for all config-files
        input_dir = os.path.join(list_of_model_result[0], dataset_midfix, 'input')
        mask_dir = os.path.join(list_of_model_result[0], dataset_midfix, 'mask')
        target_dir = os.path.join(list_of_model_result[0], dataset_midfix, 'target')
        """         Load the corresponding input-/target-/mask-files        """
        input_file_path = os.path.join(input_dir, i_file_no_ext + file_ext_model_result)
        mask_file_path = os.path.join(mask_dir, i_file_no_ext + file_ext_model_result)
        target_file_path = os.path.join(target_dir, i_file_no_ext + file_ext_model_result)
        input_array = hmisc.load_array(input_file_path)
        mask_array = hmisc.load_array(mask_file_path)
        # Lets get the target image as well...
        target_file_present = os.path.isfile(target_file_path)
        if target_file_present:
            target_array = hmisc.load_array(target_file_path)
            if 'nii' in file_ext_model_result:
                target_array = target_array.T[:, ::-1, ::-1]
        if 'nii' in file_ext_model_result:
            input_array = input_array.T[:, ::-1, ::-1]
        if 'nii' in file_ext_model_result:
            mask_array = mask_array.T[:, ::-1, ::-1]

        """         Load all the predicted images        """

        model_result_list = []
        for i_pred_dir in list_of_model_result:
            pred_file_path = os.path.join(i_pred_dir, dataset_midfix, 'pred', i_file_no_ext + file_ext_model_result)
            loaded_array = hmisc.load_array(pred_file_path)
            if 'nii' in file_ext_model_result:
                loaded_array = loaded_array.T[:, ::-1, ::-1]
            model_result_list.append(loaded_array)
        print(f'\t Loaded all the data')
        n_slice = result_n4.shape[0]
        sel_slice = n_slice // 2
        # sel_slice = 0
        result_n4_sel = result_n4[sel_slice]
        input_array_sel = input_array[sel_slice]
        mask_array_sel = mask_array[sel_slice]
        if target_file_present:
            target_array_sel = target_array[sel_slice] * mask_array_sel
            target_array_sel = harray.scale_minmax(target_array_sel)
        model_result_list_sel = [harray.scale_minmax(x[sel_slice]) for x in model_result_list]
        """         Select appropriate patches...        """
        if dataset_midfix == 'volunteer_corrected':
            # Needed for volunteer 7T data
            patch_size = 10 * 10
            dataset_name = 'volunteer'
        elif dataset_midfix == 'patient_corrected_3T':
            # Needed for patient 3T
            patch_size = 7 * 10
            dataset_name = 'patient_3T'
        elif dataset_midfix == 'patient_corrected':
            patch_size = 16 * 10
            dataset_name = 'patient'
        elif dataset_midfix == 'target_corrected':
            patch_size = 7 * 10
            dataset_name = 'test'
        else:
            dataset_name = ''
            patch_size = 10 * 10
            print('Unknown dataset name. Received: ', dataset_midfix)
        """         Equalize the images...        """
        input_array_sel = harray.scale_minmax(input_array_sel)
        result_n4_sel = harray.scale_minmax(result_n4_sel)
        # True is atleast needed for test data...?
        equalize_obj = hplotc.ImageIntensityEqualizer(reference_image=input_array_sel, image_list=[result_n4_sel] + model_result_list_sel,
                                                      patch_width=patch_size,
                                                      dynamic_thresholding=True)
        temp_images = equalize_obj.correct_image_list()
        print(f'\t Equalized all the images')
        result_n4_sel = np.array(temp_images[0])
        model_result_list_sel = temp_images[1:]
        # Creat the array to be plotted
        if target_file_present:
            plot_array = np.array([input_array_sel] + [result_n4_sel] + model_result_list_sel + [target_array_sel])
        else:
            plot_array = np.array([input_array_sel] + [result_n4_sel] + model_result_list_sel)

        _, crop_coords = harray.get_center_transformation_coords(mask_array_sel)
        plot_array = np.array([harray.apply_crop_axis(x, crop_coords=crop_coords, axis=0) for x in plot_array])
        mask_array_sel = harray.apply_crop_axis(mask_array_sel, crop_coords=crop_coords, axis=0)
        # Correct the input image...
        if dataset_midfix == 'target_corrected':
            # Because thsoe images get more ugly somehow...
            plot_array[0][plot_array[0] > 2 * equalize_obj.vmax_ref] = equalize_obj.vmax_ref
            plot_array[0] = harray.scale_minmax(plot_array[0])
        else:
            plot_array[0][plot_array[0] > equalize_obj.vmax_ref] = equalize_obj.vmax_ref
            plot_array[0] = harray.scale_minmax(plot_array[0])
        print([x.shape for x in plot_array])
        print(mask_array_sel.shape)

        plot_array = np.array([x * mask_array_sel for x in plot_array])
        # Get the homogeneity and energy list of the chosen image..
        # Collect the homogeneity and energy...
        import papers.inhomog_removal.helper_inhomog as helper_inhomog
        pred_glcm_dict = {}
        for k, v in RESULT_PATH.items():
            pred_glcm_dict = helper_inhomog.collect_glcm(v, file_name='pred_change_glcm', temp_dict=pred_glcm_dict, key=k)
        inp_glcm_dict = {}
        for k, v in RESULT_PATH.items():
            inp_glcm_dict = helper_inhomog.collect_glcm(v, file_name='input_change_glcm', temp_dict=inp_glcm_dict, key=k)
        target_glcm_dict = {}
        for k, v in RESULT_PATH.items():
            target_glcm_dict = helper_inhomog.collect_glcm(v, file_name='target_change_glcm', temp_dict=target_glcm_dict, key=k)
        # En nu... pak de juist modellen...
        # dataset_midfix - denotes dataset used..
        # list_of_model_result - contains the used models...
        random_sel = os.path.basename(list_of_model_result[0])
        import re
        random_sel = re.sub('biasfield', 'biasf', random_sel)
        random_sel = re.sub('homogeneous', 'homog', random_sel)
        random_key = f'{random_sel}_{dataset_name}'
        # Get homog/energy from input...
        print(inp_glcm_dict.keys())
        print(inp_glcm_dict[random_key]['file_list'])
        temp_file_list = inp_glcm_dict[random_key]['file_list']
        sel_index = hmisc.find_index_file(temp_file_list, i_file)
        homog_value = inp_glcm_dict[random_key]['homogeneity'][sel_index]
        energy_value = inp_glcm_dict[random_key]['energy'][sel_index]
        homogeneity_energy_list = [(homog_value, energy_value)]
        # Get homog/energy from n4..
        n4_key = f"n4_{dataset_name}"
        temp_file_list = pred_glcm_dict[n4_key]['file_list']
        sel_index = hmisc.find_index_file(temp_file_list, i_file)
        homog_value = pred_glcm_dict[n4_key]['homogeneity'][sel_index]
        energy_value = pred_glcm_dict[n4_key]['energy'][sel_index]
        homogeneity_energy_list.append((homog_value, energy_value))
        for i_model in list_of_model_result:
            sel_model_name = os.path.basename(i_model)
            sel_model_name = re.sub('biasfield', 'biasf', sel_model_name)
            sel_model_name = re.sub('homogeneous', 'homog', sel_model_name)
            model_key = f'{sel_model_name}_{dataset_name}'
            temp_file_list = pred_glcm_dict[model_key]['file_list']
            sel_index = hmisc.find_index_file(temp_file_list, i_file)
            homog_value = pred_glcm_dict[model_key]['homogeneity'][sel_index]
            energy_value = pred_glcm_dict[model_key]['energy'][sel_index]
            homogeneity_energy_list.append((homog_value, energy_value))
        if target_file_present:
            temp_file_list = target_glcm_dict[random_key]['file_list']
            sel_index = hmisc.find_index_file(temp_file_list, i_file)
            homog_value = target_glcm_dict[random_key]['homogeneity'][sel_index]
            energy_value = target_glcm_dict[random_key]['energy'][sel_index]
            homogeneity_energy_list.append((homog_value, energy_value))
        # /Test
        # Now crop the plot array in the vertical direction to save some space
        vmax_list = [[x.min(), x.max()] for x in plot_array]
        #print('vmax', vmax_list)
        #print(equalize_obj.vmax_ref)
        # The uncorrected image has no proper vmax...

        fig_obj = hplotc.ListPlot(plot_array[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))
        fig = fig_obj.figure
        height_offset = 0.05
        hplotf.add_text_box(fig, 0, 'Uncorrected', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 1, 'Reference:\nN4(ITK) algorithm', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 2, 'Corrected:\nSingle channel t-Biasfield', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 3, 'Corrected:\nSingle channel t-Image ', height_offset=height_offset, position='top', fontsize=fontsize)
        if 'patient' not in dataset_midfix:
            hplotf.add_text_box(fig, 4, 'Corrected:\nMulti channel t-Biasfield ', height_offset=height_offset, position='top', fontsize=fontsize)
            hplotf.add_text_box(fig, 5, 'Corrected:\nMulti channel t-Image ', height_offset=height_offset, position='top', fontsize=fontsize)

        if target_file_present:
            hplotf.add_text_box(fig, 6, 'Target ', height_offset=height_offset, position='top', fontsize=fontsize)

        # Now add the bottom text boxes...
        for ii, (i_hom, i_energ) in enumerate(homogeneity_energy_list):
            i_hom = "%0.2f" % i_hom
            i_energ = "%0.2f" % i_energ
            hplotf.add_text_box(fig, ii, f'H:{i_hom}        E:{i_energ}', height_offset=0, position='bottom', fontsize=fontsize)

        fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}" + '.png'), bbox_inches='tight', pad_inches=0.0)
        hplotc.close_all()
        """
        Lets do the same for cropped images
        """
        zoom_size = 100
        if dataset_midfix == 'patient_corrected':
            zoom_size = 150
        if dataset_midfix == 'target_corrected':
            zoom_size = 100
        n_midy = model_result_list_sel[0].shape[0] // 2
        n_midx = model_result_list_sel[0].shape[1] // 2
        # Redefine to get proper scaling..
        model_result_list_sel = [harray.scale_minmax(x[sel_slice]) for x in model_result_list]
        result_n4_sel = result_n4[sel_slice]  
        input_array_sel = input_array[sel_slice] 
        if target_file_present:
            target_array_sel = target_array[sel_slice] #* mask_array_sel
        # And now crop..
        crop_coords = (n_midx, n_midx, n_midy, n_midy)
        model_result_list_sel_crop = [harray.apply_crop(x, crop_coords=crop_coords, marge=zoom_size) for x in model_result_list_sel]
        result_n4_sel_crop = harray.apply_crop(result_n4_sel, crop_coords=crop_coords, marge=zoom_size)
        input_array_sel_crop = harray.apply_crop(input_array_sel, crop_coords=crop_coords, marge=zoom_size)
        if target_file_present:
            target_array_sel_crop = harray.apply_crop(target_array_sel, crop_coords=crop_coords, marge=zoom_size)
        # Equalize the images...
        equalize_obj_crop = hplotc.ImageIntensityEqualizer(reference_image=input_array_sel_crop,
                                                      image_list=[result_n4_sel_crop] + model_result_list_sel_crop,
                                                      patch_width=zoom_size,
                                                      dynamic_thresholding=True)
        temp_images = equalize_obj_crop.correct_image_list()
        result_n4_sel_crop = np.array(temp_images[0])
        model_result_list_sel_crop = temp_images[1:]
        # Creat the array to be plotted
        if target_file_present:
            plot_array_crop = np.array([input_array_sel_crop] + [result_n4_sel_crop] + model_result_list_sel_crop + [target_array_sel_crop])
        else:
            plot_array_crop = np.array([input_array_sel_crop] + [result_n4_sel_crop] + model_result_list_sel_crop)
        # Now crop the plot array in the vertical direciton to save some space
        # if dataset_midfix == 'volunteer_corrected':
        #     # Because thsoe images get more ugly somehow...
        #     plot_array_crop[0][plot_array_crop[0] > equalize_obj.vmax_ref] = equalize_obj.vmax_ref
        #     plot_array_crop[0] = harray.scale_minmax(plot_array_crop[0])

        vmax_list = [[x.min(), x.max()] for x in plot_array_crop]
        # vmax_list[0][1] = equalize_obj_crop.vmax_ref
        fig_obj = hplotc.ListPlot(plot_array_crop[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))
        fig = fig_obj.figure
        height_offset = 0.05
        hplotf.add_text_box(fig, 0, 'Uncorrected', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 1, 'Reference:\nN4(ITK) algorithm', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 2, 'Corrected:\nSingle channel t-Biasfield', height_offset=height_offset, position='top', fontsize=fontsize)
        hplotf.add_text_box(fig, 3, 'Corrected:\nSingle channel t-Image ', height_offset=height_offset, position='top', fontsize=fontsize)
        if 'patient' not in dataset_midfix:
            hplotf.add_text_box(fig, 4, 'Corrected:\nMulti channel t-Biasfield ', height_offset=height_offset, position='top', fontsize=fontsize)
            hplotf.add_text_box(fig, 5, 'Corrected:\nMulti channel t-Image ', height_offset=height_offset, position='top', fontsize=fontsize)
        if target_file_present:
            hplotf.add_text_box(fig, 6, 'Target ', height_offset=height_offset, position='top', fontsize=fontsize)
        fig.savefig(os.path.join(ddest, i_file_no_ext + f"_{sel_slice}_crop" + '.png'), bbox_inches='tight', pad_inches=0.0)
        hplotc.close_all()
