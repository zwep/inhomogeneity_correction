import json
import matplotlib.pyplot as plt
import skimage.metrics
import multiprocessing as mp
# SOURCE: https://pythonspeed.com/articles/python-multiprocessing/
import helper.misc as hmisc
import helper.array_transf as harray
import scipy.stats
import small_project.homogeneity_measure.metric_implementations as homog_metric
import os
import numpy as np
import skimage.feature
import helper.plot_class as hplotc


class CalculateMetrics:
    def __init__(self, dinput, dpred, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.dinput = dinput
        self.dpred = dpred
        self.dmask = kwargs.get('dmask', None)
        self.dtarget = kwargs.get('dtarget', None)
        # Optional mask files
        self.dmask_prostate = kwargs.get('dmask_prostate', '')
        self.dmask_fat = kwargs.get('dmask_fat', '')
        self.dmask_muscle = kwargs.get('dmask_muscle', '')
        self.file_list = sorted(os.listdir(dinput))
        # Sometimes mask files have some suffix...
        self.mask_suffix = kwargs.get('mask_suffix', '')
        self.mask_ext = kwargs.get('mask_ext', '')
        self.patch_size = kwargs.get('patch_size', None)
        self.slice_patch_size = None
        self.mid_slice = kwargs.get('mid_slice', False)
        self.shrink_pixels = kwargs.get('shrink_pixels', 0)
        # Parameters for GLCM stuff
        self.glcm_dist = kwargs.get('glcm_dist', [1, 2])
        self.max_slices = 30
        self.loaded_image = None
        self.loaded_image_slice = None
        self.loaded_target = None
        self.loaded_target_slice = None
        self.loaded_pred = None
        self.loaded_pred_slice = None
        self.loaded_mask = None
        self.loaded_mask_prostate = None
        self.loaded_mask_fat = None
        self.loaded_mask_muscle = None
        self.loaded_mask_slice = None
        self.n_slices = None
        self.n_cores = mp.cpu_count()
        self.feature_keys = kwargs.get('feature_keys', None)
        if self.feature_keys is None:
            self.feature_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    def print_features_current_slice(self):
        # Now calculate current values..
        temp_pred_slice, temp_mask_slice = harray.get_crop(self.loaded_pred_slice, self.loaded_mask_slice)
        temp_image_slice, _ = harray.get_crop(self.loaded_image_slice, self.loaded_mask_slice)
        rel_temp_dict, pred_temp_dict, input_temp_dict = self.get_glcm_slice(temp_pred_slice, temp_image_slice,
                                                                                   patch_size=self.slice_patch_size)
        # Get relative COV features
        temp_inp = temp_image_slice[temp_mask_slice != 0]
        temp_pred = temp_pred_slice[temp_mask_slice != 0]
        coeff_var_input_value = np.std(temp_inp) / np.mean(temp_inp)
        coeff_var_pred_value = np.std(temp_pred) / np.mean(temp_pred)
        coeff_var_rel_value = (coeff_var_pred_value - coeff_var_input_value) / coeff_var_input_value
        print("Relative GLCM dict ")
        hmisc.print_dict(rel_temp_dict)
        print("Relative COefv ", coeff_var_rel_value)
        return {'glcm': (rel_temp_dict, pred_temp_dict, input_temp_dict),
                'cov': (coeff_var_input_value, coeff_var_pred_value, coeff_var_rel_value)}

    def print_target_features_current_slice(self):
        if self.dtarget is not None:
            rmse_value = np.sqrt(np.mean((self.loaded_pred_slice - self.loaded_target_slice) ** 2))
            wss_value = scipy.stats.wasserstein_distance(self.loaded_pred_slice.ravel(), self.loaded_target_slice.ravel())
            ssim_target_pred = skimage.metrics.structural_similarity(self.loaded_pred_slice, self.loaded_target_slice)
            print("Wasserstein distance ", wss_value)
            print("SSIM ", ssim_target_pred)
            print("RMSE ", rmse_value)
            return rmse_value, wss_value, ssim_target_pred

    def save_current_slice(self, storage_name):
        # Will always be stored at a specific location
        if self.dtarget is not None:
            target_slice = self.loaded_target_slice
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        n = 256
        input_hist, bins_numpy = np.histogram(self.loaded_image_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
        ax[0].bar(bins_numpy[:-1], input_hist, width=1, color='b', alpha=0.5, label='input')
        ax[1].imshow(self.loaded_image_slice * self.loaded_mask_slice)
        ax[1].set_title('input')
        pred_hist, _ = np.histogram(self.loaded_pred_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
        ax[2].imshow(self.loaded_pred_slice * self.loaded_mask_slice)
        ax[2].set_title('pred')
        ax[0].bar(bins_numpy[:-1], pred_hist, width=1, color='r', alpha=0.5, label='pred')
        if self.dtarget is not None:
            target_hist, _ = np.histogram(self.loaded_target_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
            ax[0].bar(bins_numpy[:-1], target_hist, width=1, color='g', alpha=0.5, label='target')
            ax[3].imshow(self.loaded_target_slice * self.loaded_mask_slice)
            ax[3].set_title('target')
        ax[0].legend()
        fig.suptitle(storage_name)
        fig.savefig(f'/local_scratch/sharreve/temp_data_storage/{storage_name}.png')

    def load_file(self, file_index):
        sel_file = self.file_list[file_index]
        if len(self.mask_ext):
            sel_file_ext = self.mask_ext
        else:
            sel_file_ext = hmisc.get_ext(sel_file)

        sel_file_name = hmisc.get_base_name(sel_file)
        sel_img_file = os.path.join(self.dinput, sel_file)
        sel_mask_file = os.path.join(self.dmask, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_target_file = ''
        if self.dtarget is not None:
            sel_target_file = os.path.join(self.dtarget, sel_file)
        sel_mask_prostate_file = os.path.join(self.dmask_prostate, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_mask_fat_file = os.path.join(self.dmask_fat, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_mask_muscle_file = os.path.join(self.dmask_muscle, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_pred_file = os.path.join(self.dpred, sel_file)

        # Stuff...
        img_file_ext = hmisc.get_ext(sel_img_file)
        self.loaded_image = hmisc.load_array(sel_img_file)
        if 'nii' in img_file_ext:
            self.loaded_image = self.loaded_image.T[:, ::-1, ::-1]

        self.loaded_pred = hmisc.load_array(sel_pred_file)
        if 'nii' in hmisc.get_ext(sel_pred_file):
            self.loaded_pred = self.loaded_pred.T[:, ::-1, ::-1]

        if os.path.isfile(sel_target_file):
            self.loaded_target = hmisc.load_array(sel_target_file)
            if 'nii' in hmisc.get_ext(sel_target_file):
                self.loaded_target = self.loaded_target.T[:, ::-1, ::-1]
            # self.loaded_target = harray.scale_minmax(self.loaded_target, axis=(-2, -1))

        if os.path.isfile(sel_mask_file):
            self.loaded_mask = self.load_mask(sel_mask_file)
            # Type casting is needed for scaling: this ensures that we have a binary mask
            # self.loaded_mask = harray.scale_minmax(self.loaded_mask.astype(np.int8))
            self.loaded_mask = skimage.img_as_bool(self.loaded_mask).astype(int)
        if os.path.isfile(sel_mask_prostate_file):
            self.loaded_mask_prostate = skimage.img_as_bool(self.loaded_mask_prostate).astype(int)
        if os.path.isfile(sel_mask_fat_file):
            self.loaded_mask_fat = self.load_mask(sel_mask_fat_file)
            self.loaded_mask_fat = skimage.img_as_bool(self.loaded_mask_fat).astype(int)
        if os.path.isfile(sel_mask_muscle_file):
            self.loaded_mask_muscle = skimage.img_as_bool(self.loaded_mask_muscle).astype(int)

        if 'nii' in hmisc.get_ext(sel_mask_file):
            self.loaded_mask = self.loaded_mask.T[:, ::-1, ::-1]
        if 'nii' in hmisc.get_ext(sel_mask_fat_file):
            self.loaded_mask_fat = self.loaded_mask_fat.T[:, ::-1, ::-1]

        self.n_slices = self.loaded_image.shape[0]
        # self.loaded_image = harray.scale_minmax(self.loaded_image, axis=(-2, -1))
        # self.loaded_pred = harray.scale_minmax(self.loaded_pred, axis=(-2, -1))

    def set_slice(self, slice_index):
        self.loaded_mask_slice = self.loaded_mask[slice_index]
        # Some CLAHE stuff
        # Now we can do several histogram equilization things...
        import cv2
        from skimage import img_as_ubyte
        # nx, ny = self.loaded_mask_slice.shape
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
        # Load array...
        if self.loaded_mask_fat is not None:
            self.loaded_mask_slice = self.loaded_mask_slice * self.loaded_mask_fat[slice_index]
        self.loaded_image_slice = self.loaded_image[slice_index]
        self.loaded_image_slice = harray.scale_minmax(self.loaded_image_slice)
        self.loaded_image_slice = img_as_ubyte(self.loaded_image_slice)
        # self.loaded_image_slice = clahe.apply(self.loaded_image_slice)

        self.loaded_pred_slice = self.loaded_pred[slice_index]
        self.loaded_pred_slice = harray.scale_minmax(self.loaded_pred_slice)
        self.loaded_pred_slice = img_as_ubyte(self.loaded_pred_slice)
        # self.loaded_pred_slice = clahe.apply(self.loaded_pred_slice)

        self.slice_patch_size = self.get_patch_size(self.loaded_mask_slice)
        if self.loaded_target is not None:
            self.loaded_target_slice = self.loaded_target[slice_index]
            self.loaded_target_slice = harray.scale_minmax(self.loaded_target_slice)
            self.loaded_target_slice = img_as_ubyte(self.loaded_target_slice)
            # self.loaded_target_slice = clahe.apply(self.loaded_target_slice)
        #
        # # Equalizing the images...
        # equalize_obj = hplotc.ImageIntensityEqualizer(reference_image=self.loaded_image_slice, image_list=[self.loaded_pred_slice],
        #                                               patch_width=self.patch_size,
        #                                               dynamic_thresholding=True)
        # temp_images = equalize_obj.correct_image_list()
        # self.loaded_pred_slice = np.array(temp_images[0])
        # self.loaded_image_slice[self.loaded_image_slice > equalize_obj.vmax_ref] = equalize_obj.vmax_ref
        # self.loaded_image_slice = harray.scale_minmax(self.loaded_image_slice)

        # Adding this... again...
        # Wil ik dit echt....? -> Geen idee voor nu 04/07/2022
        # import cv2
        # nx, ny = self.loaded_mask_slice.shape
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
        # self.loaded_image_slice = clahe.apply(self.loaded_image_slice)
        # self.loaded_pred_slice = clahe.apply(self.loaded_pred_slice)
        # if self.dtarget is not None:
        #     self.loaded_target_slice = clahe.apply(self.loaded_target_slice)
        # Weg gehaald voor nu... 12/07/2022

    @staticmethod
    def load_mask(x):
        loaded_mask = hmisc.load_array(x)
        if loaded_mask.ndim == 2:
            loaded_mask = loaded_mask[None]
        return loaded_mask

    def run_features(self, debug=False):
        # Do a complete run...fixed to to a patch-based evaluation
        glcm_rel = []
        glcm_input = []
        glcm_pred = []
        coef_var_rel = []
        coef_var_input = []
        coef_var_pred = []
        slice_list = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            # Somewhere we suddenly assume a loaded_mask to exist....
            # It should work without as well...
            if self.mid_slice:
                range_of_slices = [self.n_slices // 2]
            else:
                if self.n_slices < 20:
                    range_of_slices = range(self.n_slices)
                else:
                    range_of_slices = range(10, 20)

            for slice_index in range_of_slices:
                print(f'Running slice {slice_index} / {self.n_slices}', end='\r')
                self.set_slice(slice_index)
                self.loaded_mask_slice = harray.shrink_image(self.loaded_mask_slice, self.shrink_pixels)
                self.loaded_image_slice = np.ma.masked_array(self.loaded_image_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_pred_slice = np.ma.masked_array(self.loaded_pred_slice, mask=1 - self.loaded_mask_slice)
                center_mask = harray.create_random_center_mask(self.loaded_image_slice.shape, random=False)

                mean_pred = np.mean(self.loaded_pred_slice[center_mask==1])
                mean_input = np.mean(self.loaded_image_slice[center_mask==1])
                pred_to_input_scaling = mean_input / mean_pred
                # Mean intensity value is now similar between prediction and target
                self.loaded_image_slice = self.loaded_image_slice * 1.
                self.loaded_pred_slice = self.loaded_pred_slice * pred_to_input_scaling

                # Get GLCM features
                # Cropping is needed to avoid pathces without much info...
                # 04-7-22: removed this because we are using masked valued
                # temp_pred_slice, temp_mask_slice = harray.get_crop(self.loaded_pred_slice, self.loaded_mask_slice)
                # temp_image_slice, _ = harray.get_crop(self.loaded_image_slice, self.loaded_mask_slice)
                rel_temp_dict, pred_temp_dict, input_temp_dict = self.get_glcm_slice(self.loaded_pred_slice, self.loaded_image_slice, patch_size=self.slice_patch_size)
                glcm_rel.append(rel_temp_dict)
                glcm_input.append(input_temp_dict)
                glcm_pred.append(pred_temp_dict)
                # Get relative COV features
                coeff_var_input_value = np.std(self.loaded_image_slice) / np.mean(self.loaded_image_slice)
                coeff_var_pred_value = np.std(self.loaded_pred_slice) / np.mean(self.loaded_pred_slice)
                coeff_var_rel_value = (coeff_var_pred_value - coeff_var_input_value) / coeff_var_input_value
                if debug:
                    print("Min/Mean/Median/Max mask ", harray.get_minmeanmediammax(self.loaded_mask_slice))
                    print("Min/Mean/Median/Max image ", harray.get_minmeanmediammax(self.loaded_image_slice))
                    print("Min/Mean/Median/Max pred ", harray.get_minmeanmediammax(self.loaded_pred_slice))
                    print("Relative GLCM dict ", rel_temp_dict)
                    print("Relative COefv ", coeff_var_rel_value)
                coef_var_rel.append(coeff_var_rel_value)
                coef_var_input.append(coeff_var_input_value)
                coef_var_pred.append(coeff_var_pred_value)
                slice_list.append(range_of_slices)
        return glcm_rel, glcm_input, glcm_pred, coef_var_rel, coef_var_input, coef_var_pred, slice_list

    def run_features_target(self):
        # Do a complete run...fixed to to a patch-based evaluation
        glcm_rel = []
        glcm_target = []
        coefv_target = []
        coefv_target_rel = []
        RMSE_list = []
        WSS_distance = []
        SSIM_list = []
        slice_list = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            # Somewhere we suddenly assume a loaded_mask to exist....
            # It should work without as well...
            if self.mid_slice:
                range_of_slices = [self.n_slices // 2]
            else:
                if self.n_slices < 20:
                    range_of_slices = range(self.n_slices)
                else:
                    range_of_slices = range(10, 20)

            for slice_index in range_of_slices:
                print(f'Running slice {slice_index} / {self.n_slices}', end='\r')
                self.set_slice(slice_index)
                # Shrink the mask... and check it out...?
                self.loaded_mask_slice = harray.shrink_image(self.loaded_mask_slice, self.shrink_pixels)
                # ...
                self.loaded_target_slice = np.ma.masked_array(self.loaded_target_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_image_slice = np.ma.masked_array(self.loaded_image_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_pred_slice = np.ma.masked_array(self.loaded_pred_slice, mask=1 - self.loaded_mask_slice)

                center_mask = harray.create_random_center_mask(self.loaded_image_slice.shape, random=False)
                # Make sure target and input image are similar in scaling
                # Create a mask in the center from which we get values..
                mean_pred = np.mean(self.loaded_pred_slice[center_mask==1])
                mean_input = np.mean(self.loaded_image_slice[center_mask==1])
                mean_target = np.mean(self.loaded_target_slice[center_mask==1])
                pred_to_target_scaling = mean_target / mean_pred
                input_to_target_scaling = mean_target / mean_input
                # Mean intensity value is now similar between prediction and target
                self.loaded_image_slice = self.loaded_image_slice * input_to_target_scaling
                self.loaded_pred_slice = self.loaded_pred_slice * pred_to_target_scaling
                self.loaded_target_slice = self.loaded_target_slice * 1.

                # if self.debug:
                #     print("Image characteristics...")
                #     print("Loaded image slice ", harray.get_minmeanmediammax(self.loaded_image_slice), self.loaded_image_slice.dtype)
                #     print("Loaded pred slice ", harray.get_minmeanmediammax(self.loaded_pred_slice), self.loaded_pred_slice.dtype)
                #     print("Loaded target slice ", harray.get_minmeanmediammax(self.loaded_target_slice), self.loaded_target_slice.dtype)

                # Calculate histograms
                slice_hist_target, _ = np.histogram(self.loaded_target_slice.ravel(), bins=256, range=(0, 255),
                                                    density=True)
                slice_hist_pred, _ = np.histogram(self.loaded_pred_slice.ravel(), bins=256, range=(0, 255),
                                                  density=True)

                # Calculate all the features..
                rel_temp_dict, target_temp_dict, input_temp_dict = self.get_glcm_slice(self.loaded_target_slice, self.loaded_image_slice, patch_size=self.slice_patch_size)
                coeff_var_target_value = np.std(self.loaded_target_slice) / np.mean(self.loaded_target_slice)
                coeff_var_input_value = np.std(self.loaded_image_slice) / np.mean(self.loaded_image_slice)
                coeff_var_rel_value = (coeff_var_target_value - coeff_var_input_value) / coeff_var_input_value

                rmse_value = np.sqrt(np.mean((self.loaded_pred_slice - self.loaded_target_slice) ** 2))
                wss_value = scipy.stats.wasserstein_distance(slice_hist_pred, slice_hist_target)
                ssim_target_pred = skimage.metrics.structural_similarity(self.loaded_pred_slice, self.loaded_target_slice)

                if self.debug:
                    print("Scaling values ", pred_to_target_scaling)
                    print("Min/Mean/Median/Max mask ", harray.get_minmeanmediammax(self.loaded_mask_slice))
                    print("Min/Mean/Median/Max image ", harray.get_minmeanmediammax(self.loaded_image_slice))
                    print("Min/Mean/Median/Max target ", harray.get_minmeanmediammax(self.loaded_target_slice))
                    print("Min/Mean/Median/Max pred ", harray.get_minmeanmediammax(self.loaded_pred_slice))
                    print("Wasserstein distance ", wss_value)
                    print("SSIM ", ssim_target_pred)
                    print("RMSE ", rmse_value)
                coefv_target.append(coeff_var_target_value)
                coefv_target_rel.append(coeff_var_rel_value)
                glcm_target.append(target_temp_dict)
                glcm_rel.append(rel_temp_dict)
                RMSE_list.append(rmse_value)
                SSIM_list.append(ssim_target_pred)
                WSS_distance.append(wss_value)
                slice_list.append(range_of_slices)
        return coefv_target_rel, coefv_target, glcm_rel, glcm_target, RMSE_list, SSIM_list, WSS_distance, slice_list

    def get_patch_size(self, x):
        if self.patch_size is None:
            patch_size = min(x.shape) // 3
        elif self.patch_size == 'max':
            patch_size = min(x.shape)
        else:
            patch_size = self.patch_size
        return patch_size

    def get_glcm_slice(self, image_a, image_b, patch_size):
        print("Get GLCM slice ... ")
        print(f"image A shape {image_a.shape} ")
        print(f"image B shape {image_b.shape} ")
        print(f"patch size {patch_size} ")
        glcm_obj_a = homog_metric.get_glcm_patch_object(image_a, patch_size=patch_size, glcm_dist=self.glcm_dist)
        glcm_obj_b = homog_metric.get_glcm_patch_object(image_b, patch_size=patch_size, glcm_dist=self.glcm_dist)
        feature_dict_rel = {}
        feature_dict_a = {}
        feature_dict_b = {}
        n_patches = float(len(glcm_obj_a))
        for patch_obj_a, patch_obj_b in zip(glcm_obj_a, glcm_obj_b):
            for i_feature in self.feature_keys:
                _ = feature_dict_rel.setdefault(i_feature, 0)
                _ = feature_dict_a.setdefault(i_feature, 0)
                _ = feature_dict_b.setdefault(i_feature, 0)
                feature_value_a = skimage.feature.greycoprops(patch_obj_a, i_feature)
                feature_value_b = skimage.feature.greycoprops(patch_obj_b, i_feature)
                rel_change = (feature_value_a - feature_value_b) / feature_value_b
                feature_dict_rel[i_feature] += np.mean(rel_change) / n_patches
                feature_dict_a[i_feature] += np.mean(feature_value_a) / n_patches
                feature_dict_b[i_feature] += np.mean(feature_value_b) / n_patches

        return feature_dict_rel, feature_dict_a, feature_dict_b
