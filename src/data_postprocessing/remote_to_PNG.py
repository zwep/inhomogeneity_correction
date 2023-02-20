"""
Ugh.. niftis..
Gimme some PNGs
"""

import os
import helper.misc as hmisc
import helper.plot_class as hplotc
import matplotlib.pyplot as plt
import re
import helper.array_transf as harray
import numpy as np

ddata = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

for d, _, file_list in os.walk(ddata):
    filter_file_list = [x for x in file_list if x.endswith('nii.gz')]
    # if ('target_corrected' in d):# and (('single_biasfield' in d) or ('single_homogeneous' in d)):
    if (d.endswith('pred') or d.endswith('input')) and len(filter_file_list):
        substitute_dir = os.path.basename(d)
        dest_dir = d + '_PNG'
        dest_dir_hist = d + 'hist_PNG'
        d_mask = re.sub(f'/{substitute_dir}', '/mask', d)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        if not os.path.isdir(dest_dir_hist):
            os.makedirs(dest_dir_hist)
        print(d, dest_dir)
        # Just do max 10..
        for i_file in filter_file_list:#[:10]:
            base_name = hmisc.get_base_name(i_file)
            file_path = os.path.join(d, i_file)
            mask_file_path = os.path.join(d_mask, i_file)
            print(i_file)
            dest_file_path = os.path.join(dest_dir, base_name + '.png')
            dest_file_path_hist = os.path.join(dest_dir_hist, base_name + '.png')
            loaded_array = hmisc.load_array(file_path)
            loaded_mask = hmisc.load_array(mask_file_path)
            # Change order because NIFITs
            loaded_array = loaded_array.T[:, ::-1, ::-1]
            loaded_mask = loaded_mask.T[:, ::-1, ::-1]
            n_slice = loaded_array.shape[0]
            sel_slice = n_slice // 2
            sel_img = loaded_array[sel_slice]
            sel_mask = loaded_mask[sel_slice]
            import skimage
            sel_mask = skimage.img_as_bool(sel_mask).astype(int)
            sel_mask = harray.shrink_image(sel_mask, 0)
            fig_obj = hplotc.ListPlot(sel_img * sel_mask, ax_off=True)
            fig_obj.figure.savefig(dest_file_path, bbox_inches='tight', pad_inches=0.0)
            # fig, ax = plt.subplots()
            # _ = ax.hist([ > 1].ravel(), bins=256, range=(0, 255), density=True)
            # fig.savefig(dest_file_path_hist)
            hplotc.close_all()
