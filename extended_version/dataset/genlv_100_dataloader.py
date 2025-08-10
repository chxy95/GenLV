import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

import cv2
import sys
import random

import dataset.util as util
from dataset.add_degradation_various import *
from dataset.image_operators import *
from dataset.x_distortion import *
from basicsr.utils.matlab_functions import imresize

def add_x_distortion_two_images(img_gt1, img_gt2, deg_type):
    # np.uint8, BGR
    x_distortion_dict = distortions_dict
    severity = random.choice([1, 2, 3, 4, 5])
    deg_type = random.choice(x_distortion_dict[deg_type])
    
    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2RGB)
    img_gt2 = cv2.cvtColor(img_gt2, cv2.COLOR_BGR2RGB)
        
    img_lq1 = globals()[deg_type](img_gt1, severity)
    img_lq2 = globals()[deg_type](img_gt2, severity)

    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_RGB2BGR)
    img_gt2 = cv2.cvtColor(img_gt2, cv2.COLOR_RGB2BGR)
    img_lq1 = cv2.cvtColor(img_lq1, cv2.COLOR_RGB2BGR)
    img_lq2 = cv2.cvtColor(img_lq2, cv2.COLOR_RGB2BGR)
    
    return img_lq1, img_lq2, img_gt1, img_gt2, deg_type

def calculate_operators_two_images(img_gt1, img_gt2, deg_type):
    # np.uint8
    if deg_type == 'Laplacian':
        img_lq1 = img_gt1.copy()
        img_gt1 = Laplacian_edge_detector_uint8(img_gt1)
        img_lq2 = img_gt2.copy()
        img_gt2 = Laplacian_edge_detector_uint8(img_gt2)
    elif deg_type == 'Canny':
        img_lq1 = img_gt1.copy()
        img_gt1 = Canny_edge_detector_uint8(img_gt1)
        img_lq2 = img_gt2.copy()
        img_gt2 = Canny_edge_detector_uint8(img_gt2)
    # check zero images.
    if np.mean(img_gt1).astype(np.float16) == 0 or np.mean(img_gt2).astype(np.float16) == 0:
        if np.mean(img_gt1).astype(np.float16) == 0 and np.mean(img_gt2).astype(np.float16) == 0:
            print(deg_type, 'prompt&query zero images.')
            img_gt1 = img_lq1.copy()
            img_gt2 = img_lq2.copy()
        elif np.mean(img_gt1).astype(np.float16) == 0:
            print(deg_type, 'prompt gt zero image.')
            img_gt1 = img_gt2.copy()
            img_lq1 = img_lq2.copy()
        elif np.mean(img_gt2).astype(np.float16) == 0:
            print(deg_type, 'query gt zero image.')
            img_gt2 = img_gt1.copy()
            img_lq2 = img_lq1.copy()
    return img_lq1, img_lq2, img_gt1, img_gt2 

def add_degradation_two_images(img_gt1, img_gt2, deg_type):
    # np.float32
    if deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq1 = add_rain(img_gt1, value=value)
        value = random.uniform(40, 200)
        img_lq2 = add_rain(img_gt2, value=value)
    elif deg_type == 'Ringing':
        img_lq1 = add_ringing(img_gt1)
        img_lq2 = add_ringing(img_gt2)
    elif deg_type == 'r_l':
        img_lq1 = r_l(img_gt1)
        img_lq2 = r_l(img_gt2)
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_lq1 = inpainting(img_gt1, l_num=l_num, l_thick=l_thick)
        img_lq2 = inpainting(img_gt2, l_num=l_num, l_thick=l_thick)
    elif deg_type == 'mosaic':
        img_lq1 = mosaic_CFA_Bayer(img_gt1)
        img_lq2 = mosaic_CFA_Bayer(img_gt2)
    elif deg_type == 'SRx2':
        H, W, _ = img_gt1.shape
        img_lq1 = imresize(img_gt1, 1/2)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)
        img_lq2 = imresize(img_gt2, 1/2)
        img_lq2 = cv2.resize(img_lq2, (W, H), interpolation=cv2.INTER_CUBIC)
    elif deg_type == 'SRx4':
        H, W, _ = img_gt1.shape
        img_lq1 = imresize(img_gt1, 1/4)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)
        img_lq2 = imresize(img_gt2, 1/4)
        img_lq2 = cv2.resize(img_lq2, (W, H), interpolation=cv2.INTER_CUBIC)

    elif deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq1 = add_Gaussian_noise(img_gt1, level=level)
        level = random.uniform(10, 50)
        img_lq2 = add_Gaussian_noise(img_gt2, level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq1 = iso_GaussianBlur(img_gt1, window=15, sigma=sigma)
        sigma = random.uniform(2, 4)
        img_lq2 = iso_GaussianBlur(img_gt2, window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq1 = add_JPEG_noise(img_gt1, level=level)
        level = random.randint(10, 40)
        img_lq2 = add_JPEG_noise(img_gt2, level=level)
    elif deg_type == 'Resize':
        img_lq1 = add_resize(img_gt1)
        img_lq2 = add_resize(img_gt2)
    elif deg_type == 'SPNoise':
        img_lq1 = add_sp_noise(img_gt1)
        img_lq2 = add_sp_noise(img_gt2)
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq1 = low_light(img_gt1, lum_scale=lum_scale)
        img_lq2 = low_light(img_gt2, lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq1 = add_Poisson_noise(img_gt1, level=2)
        img_lq2 = add_Poisson_noise(img_gt2, level=2)
    elif deg_type == 'gray':
        img_lq1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2GRAY)
        img_lq1 = np.expand_dims(img_lq1, axis=2)
        img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
        img_lq2 = cv2.cvtColor(img_gt2, cv2.COLOR_BGR2GRAY)
        img_lq2 = np.expand_dims(img_lq2, axis=2)
        img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)
    elif deg_type == 'None':
        img_lq1 = img_gt1
        img_lq2 = img_gt2
    else:
        print('Error!', '-', deg_type, '-')
        exit()

    img_lq1 = np.clip(img_lq1*255, 0, 255).round().astype(np.uint8)
    img_lq1 = img_lq1.astype(np.float32)/255.0
    
    img_lq2 = np.clip(img_lq2*255, 0, 255).round().astype(np.uint8)
    img_lq2 = img_lq2.astype(np.float32)/255.0
    
    img_gt1 = np.clip(img_gt1*255, 0, 255).round().astype(np.uint8)
    img_gt1 = img_gt1.astype(np.float32)/255.0
    
    img_gt2 = np.clip(img_gt2*255, 0, 255).round().astype(np.uint8)
    img_gt2 = img_gt2.astype(np.float32)/255.0

    return img_lq1, img_lq2, img_gt1, img_gt2

class DatasetPrompt_Train(Dataset):
    def __init__(self, dataset_path, input_size, 
                 meta_info_file=None, # notice 'data/meta_info_ImageNet_GT.txt'
                 # Restoration: 17 (-1)
                 Derain_path='data/GenLV_Processed_Data/Restoration/Derain', 
                 RainDrop_path='data/GenLV_Processed_Data/Restoration/RainDrop', 
                 MarineSnowRemoval_path='data/GenLV_Processed_Data/Restoration/MarineSnowRemoval', # notice directory location
                 ReflectionRemoval_path='data/GenLV_Processed_Data/Restoration/ReflectionRemoval', 
                 ShadowRemovalISTD_path='data/GenLV_Processed_Data/Restoration/ShadowRemovalISTD',
                 ShadowRemovalSRD_path='data/GenLV_Processed_Data/Restoration/ShadowRemovalSRD', 
                 CloudRemoval_path='data/GenLV_Processed_Data/Restoration/CloudRemoval', 
                 WatermarkRemoval_path='data/GenLV_Processed_Data/Restoration/WatermarkRemoval', 
                 RealLLSR_path='data/GenLV_Processed_Data/Restoration/RealLLSR', 
                 UDCPoled_path='data/GenLV_Processed_Data/Restoration/UDCPoled',
                 UDCToled_path='data/GenLV_Processed_Data/Restoration/UDCToled', 
                 Dehaze_path='data/GenLV_Processed_Data/Restoration/Dehaze', 
                 Demoireing_path='data/GenLV_Processed_Data/Restoration/Demoireing', 
                 DustRemoval_path='data/GenLV_Processed_Data/Restoration/DustRemoval', 
                 Desnow_path='data/GenLV_Processed_Data/Restoration/Desnow', 
                 FlareRemoval_path='data/GenLV_Processed_Data/Restoration/FlareRemoval',
                 HighlightRemoval_path='data/GenLV_Processed_Data/Restoration/HighlightRemoval', 
                 # Enhancement: 14 (+2)
                 LowLight_path='data/GenLV_Processed_Data/Enhancement/LowLight', 
                 BacklitEnhance_path='data/GenLV_Processed_Data/Enhancement/BacklitEnhance', 
                 ExpoCorrect_path='data/GenLV_Processed_Data/Enhancement/ExpoCorrect', 
                 ISP_path='data/GenLV_Processed_Data/Enhancement/ISP',
                 LocalLapFilter_path='data/GenLV_Processed_Data/Enhancement/LocalLapFilter', 
                 PhotoRetouch_path='data/GenLV_Processed_Data/Enhancement/PhotoRetouch', 
                 RenderBokeh_path='data/GenLV_Processed_Data/Enhancement/RenderBokeh', 
                 HistoEqual_path='data/GenLV_Processed_Data/Enhancement/HistoEqual', 
                 WhiteBalance_path='data/GenLV_Processed_Data/Enhancement/WhiteBalance',
                 ColorCorrect_path='data/GenLV_Processed_Data/Enhancement/ColorCorrect', 
                 VignettingRemoval_path='data/GenLV_Processed_Data/Enhancement/VignettingRemoval', 
                 MultiScaleTM_path='data/GenLV_Processed_Data/Enhancement/MultiScaleTM', 
                 SDRHDR_path='data/GenLV_Processed_Data/Enhancement/SDRHDR', # notice directory name
                 InstagramFilter_path='data/GenLV_Processed_Data/Enhancement/InstagramFilter', # notice directory name
                 # FeatureExtra: 6
                 DepthEstimate_path='data/GenLV_Processed_Data/FeatureExtra/DepthEstimate', 
                 PercepEdgeDetect_path='data/GenLV_Processed_Data/FeatureExtra/PercepEdgeDetect', 
                 SaliencyObject_path='data/GenLV_Processed_Data/FeatureExtra/SaliencyObject',
                 HoughLine_path='data/GenLV_Processed_Data/FeatureExtra/HoughLine', 
                 Normal_path='data/GenLV_Processed_Data/FeatureExtra/Normal', # notice process
                 HEDBoundary_path='data/GenLV_Processed_Data/FeatureExtra/HEDBoundary', 
                 # Stylization: 18
                 PencilDrawing_path='data/GenLV_Processed_Data/Stylization/PencilDrawing', 
                 Photographic_path='data/GenLV_Processed_Data/Stylization/Photographic', 
                 RTV_path='data/GenLV_Processed_Data/Stylization/RTV',
                 Cloisonnism_path='data/GenLV_Processed_Data/Stylization/Cloisonnism', 
                 Divisionism_path='data/GenLV_Processed_Data/Stylization/Divisionism', 
                 Fauvism_path='data/GenLV_Processed_Data/Stylization/Fauvism', 
                 Vermeer_path='data/GenLV_Processed_Data/Stylization/Vermeer', 
                 JOJO_path='data/GenLV_Processed_Data/Stylization/JOJO', 
                 Raphael_path='data/GenLV_Processed_Data/Stylization/Raphael',
                 Modernism_path='data/GenLV_Processed_Data/Stylization/Modernism', 
                 Monet_path='data/GenLV_Processed_Data/Stylization/Monet', 
                 NeoImpressionism_path='data/GenLV_Processed_Data/Stylization/NeoImpressionism', 
                 PopArt_path='data/GenLV_Processed_Data/Stylization/PopArt', 
                 Ukiyoe_path='data/GenLV_Processed_Data/Stylization/Ukiyoe', 
                 VanGogh_path='data/GenLV_Processed_Data/Stylization/VanGogh',
                 Tuner_path='data/GenLV_Processed_Data/Stylization/Tuner', 
                 Regionalism_path='data/GenLV_Processed_Data/Stylization/Regionalism', 
                 Impressionism_path='data/GenLV_Processed_Data/Stylization/Impressionism', 
                 # Special Data: 8 (+3)
                 Face_path='data/GenLV_Processed_Data/Restoration/Face', 
                 Infrared_path='data/GenLV_Processed_Data/Restoration/Infrared', 
                 CT_path='data/GenLV_Processed_Data/Restoration/CT', 
                 MRI_path='data/GenLV_Processed_Data/Restoration/MRI', 
                 Satellite_path='data/GenLV_Processed_Data/Restoration/Satellite', 
                 Weather_path='data/SuperBench_Processd_Data/Restoration/Weather', 
                 FluidFlow_path='data/SuperBench_Processd_Data/Restoration/FluidFlow', 
                 Cosmology_path='data/SuperBench_Processd_Data/Restoration/Cosmology', 
                 data_len=None, tasks_version=0
                 ):
        
        self.data_len = data_len
        self.tasks_version = tasks_version
        
        np.random.seed(5)
        self.gt_size = input_size
        
        # Base dataset
        if meta_info_file is not None:
            with open(meta_info_file, 'r') as fin:
                self.paths_base_gt = [os.path.join(dataset_path, line.split(' ')[0]) for line in fin]
        else: self.paths_base_gt, _ = util.get_image_paths('img', dataset_path)
        self.paths_base_len = len(self.paths_base_gt)


        ##########################################################################
        # Restoration
        # Derain dataset (Rain13K)
        if Derain_path is not None:
            self.dataset_path_gt_Derain = os.path.join(Derain_path, 'Train_256/target')
            self.dataset_path_lq_Derain = os.path.join(Derain_path, 'Train_256/input')
            self.paths_gt_Derain, self.sizes_gt_Derain = util.get_image_paths('img', self.dataset_path_gt_Derain)
            self.paths_lq_Derain, self.sizes_lq_Derain = util.get_image_paths('img', self.dataset_path_lq_Derain)
            self.paths_Derain_len = len(self.paths_lq_Derain)
            sorted(self.paths_gt_Derain)
            sorted(self.paths_lq_Derain)

        # RainDrop dataset 
        if RainDrop_path is not None:
            self.dataset_path_gt_RainDrop = os.path.join(RainDrop_path, 'Train_256/target')
            self.dataset_path_lq_RainDrop = os.path.join(RainDrop_path, 'Train_256/input')
            self.paths_gt_RainDrop, self.sizes_gt_RainDrop = util.get_image_paths('img', self.dataset_path_gt_RainDrop)
            self.paths_lq_RainDrop, self.sizes_lq_RainDrop = util.get_image_paths('img', self.dataset_path_lq_RainDrop)
            self.paths_RainDrop_len = len(self.paths_lq_RainDrop)
            sorted(self.paths_gt_RainDrop)
            sorted(self.paths_lq_RainDrop)

        # MarineSnowRemoval dataset
        if MarineSnowRemoval_path is not None:
            self.dataset_path_gt_MarineSnowRemoval = os.path.join(MarineSnowRemoval_path, 'Train_256/target')
            self.dataset_path_lq_MarineSnowRemoval = os.path.join(MarineSnowRemoval_path, 'Train_256/input')
            self.paths_gt_MarineSnowRemoval, self.sizes_gt_MarineSnowRemoval = util.get_image_paths('img', self.dataset_path_gt_MarineSnowRemoval)
            self.paths_lq_MarineSnowRemoval, self.sizes_lq_MarineSnowRemoval = util.get_image_paths('img', self.dataset_path_lq_MarineSnowRemoval)
            self.paths_MarineSnowRemoval_len = len(self.paths_lq_MarineSnowRemoval)
            sorted(self.paths_gt_MarineSnowRemoval)
            sorted(self.paths_lq_MarineSnowRemoval)

        # ReflectionRemoval dataset
        if ReflectionRemoval_path is not None:
            self.dataset_path_gt_ReflectionRemoval = os.path.join(ReflectionRemoval_path, 'Train_256/target')
            self.dataset_path_lq_ReflectionRemoval = os.path.join(ReflectionRemoval_path, 'Train_256/input')
            self.paths_gt_ReflectionRemoval, self.sizes_gt_ReflectionRemoval = util.get_image_paths('img', self.dataset_path_gt_ReflectionRemoval)
            self.paths_lq_ReflectionRemoval, self.sizes_lq_ReflectionRemoval = util.get_image_paths('img', self.dataset_path_lq_ReflectionRemoval)
            self.paths_ReflectionRemoval_len = len(self.paths_lq_ReflectionRemoval)
            sorted(self.paths_gt_ReflectionRemoval)
            sorted(self.paths_lq_ReflectionRemoval)

        # ShadowRemovalISTD dataset
        if ShadowRemovalISTD_path is not None:
            self.dataset_path_gt_ShadowRemovalISTD = os.path.join(ShadowRemovalISTD_path, 'Train_256/target')
            self.dataset_path_lq_ShadowRemovalISTD = os.path.join(ShadowRemovalISTD_path, 'Train_256/input')
            self.paths_gt_ShadowRemovalISTD, self.sizes_gt_ShadowRemovalISTD = util.get_image_paths('img', self.dataset_path_gt_ShadowRemovalISTD)
            self.paths_lq_ShadowRemovalISTD, self.sizes_lq_ShadowRemovalISTD = util.get_image_paths('img', self.dataset_path_lq_ShadowRemovalISTD)
            self.paths_ShadowRemovalISTD_len = len(self.paths_lq_ShadowRemovalISTD)
            sorted(self.paths_gt_ShadowRemovalISTD)
            sorted(self.paths_lq_ShadowRemovalISTD)
        
        # ShadowRemovalSRD dataset
        if ShadowRemovalSRD_path is not None:
            self.dataset_path_gt_ShadowRemovalSRD = os.path.join(ShadowRemovalSRD_path, 'Train_256/target')
            self.dataset_path_lq_ShadowRemovalSRD = os.path.join(ShadowRemovalSRD_path, 'Train_256/input')
            self.paths_gt_ShadowRemovalSRD, self.sizes_gt_ShadowRemovalSRD = util.get_image_paths('img', self.dataset_path_gt_ShadowRemovalSRD)
            self.paths_lq_ShadowRemovalSRD, self.sizes_lq_ShadowRemovalSRD = util.get_image_paths('img', self.dataset_path_lq_ShadowRemovalSRD)
            self.paths_ShadowRemovalSRD_len = len(self.paths_lq_ShadowRemovalSRD)
            sorted(self.paths_gt_ShadowRemovalSRD)
            sorted(self.paths_lq_ShadowRemovalSRD)

        # CloudRemoval dataset 
        if CloudRemoval_path is not None:
            self.dataset_path_gt_CloudRemoval = os.path.join(CloudRemoval_path, 'Train_256/target')
            self.dataset_path_lq_CloudRemoval = os.path.join(CloudRemoval_path, 'Train_256/input')
            self.paths_gt_CloudRemoval, self.sizes_gt_CloudRemoval = util.get_image_paths('img', self.dataset_path_gt_CloudRemoval)
            self.paths_lq_CloudRemoval, self.sizes_lq_CloudRemoval = util.get_image_paths('img', self.dataset_path_lq_CloudRemoval)
            self.paths_CloudRemoval_len = len(self.paths_lq_CloudRemoval)
            sorted(self.paths_gt_CloudRemoval)
            sorted(self.paths_lq_CloudRemoval)

        # WatermarkRemoval dataset
        if WatermarkRemoval_path is not None:
            self.dataset_path_gt_WatermarkRemoval = os.path.join(WatermarkRemoval_path, 'Train_256/target')
            self.dataset_path_lq_WatermarkRemoval = os.path.join(WatermarkRemoval_path, 'Train_256/input')
            self.paths_gt_WatermarkRemoval, self.sizes_gt_WatermarkRemoval = util.get_image_paths('img', self.dataset_path_gt_WatermarkRemoval)
            self.paths_lq_WatermarkRemoval, self.sizes_lq_WatermarkRemoval = util.get_image_paths('img', self.dataset_path_lq_WatermarkRemoval)
            self.paths_WatermarkRemoval_len = len(self.paths_lq_WatermarkRemoval)
            sorted(self.paths_gt_WatermarkRemoval)
            sorted(self.paths_lq_WatermarkRemoval)

        # RealLLSR dataset - Notice loader
        if RealLLSR_path is not None:
            self.dataset_path_gt_RealLLSR = os.path.join(RealLLSR_path, 'Train_256/target')
            self.dataset_path_lq_RealLLSR = os.path.join(RealLLSR_path, 'Train_256/input')
            self.paths_gt_RealLLSR, self.sizes_gt_RealLLSR = util.get_image_paths('img', self.dataset_path_gt_RealLLSR)
            self.paths_lq_RealLLSR, self.sizes_lq_RealLLSR = util.get_image_paths('img', self.dataset_path_lq_RealLLSR)
            self.paths_gt_RealLLSR_len = len(self.paths_gt_RealLLSR)
            self.paths_lq_RealLLSR_len = len(self.paths_lq_RealLLSR)
            sorted(self.paths_lq_RealLLSR)

        # UDCPoled dataset
        if UDCPoled_path is not None:
            self.dataset_path_gt_UDCPoled = os.path.join(UDCPoled_path, 'Train_256/target')
            self.dataset_path_lq_UDCPoled = os.path.join(UDCPoled_path, 'Train_256/input')
            self.paths_gt_UDCPoled, self.sizes_gt_UDCPoled = util.get_image_paths('img', self.dataset_path_gt_UDCPoled)
            self.paths_lq_UDCPoled, self.sizes_lq_UDCPoled = util.get_image_paths('img', self.dataset_path_lq_UDCPoled)
            self.paths_UDCPoled_len = len(self.paths_lq_UDCPoled)
            sorted(self.paths_gt_UDCPoled)
            sorted(self.paths_lq_UDCPoled)
        
        # UDCToled dataset
        if UDCToled_path is not None:
            self.dataset_path_gt_UDCToled = os.path.join(UDCToled_path, 'Train_256/target')
            self.dataset_path_lq_UDCToled = os.path.join(UDCToled_path, 'Train_256/input')
            self.paths_gt_UDCToled, self.sizes_gt_UDCToled = util.get_image_paths('img', self.dataset_path_gt_UDCToled)
            self.paths_lq_UDCToled, self.sizes_lq_UDCToled = util.get_image_paths('img', self.dataset_path_lq_UDCToled)
            self.paths_UDCToled_len = len(self.paths_lq_UDCToled)
            sorted(self.paths_gt_UDCToled)
            sorted(self.paths_lq_UDCToled)

        # Dehaze dataset - Notice loader
        if Dehaze_path is not None:
            self.dataset_path_gt_Dehaze = os.path.join(Dehaze_path, 'Train_256/target')
            self.dataset_path_lq_Dehaze = os.path.join(Dehaze_path, 'Train_256/input')
            self.paths_gt_Dehaze, self.sizes_gt_Dehaze = util.get_image_paths('img', self.dataset_path_gt_Dehaze)
            self.paths_lq_Dehaze, self.sizes_lq_Dehaze = util.get_image_paths('img', self.dataset_path_lq_Dehaze)
            self.paths_Dehaze_len = len(self.paths_lq_Dehaze)
            sorted(self.paths_lq_Dehaze)

        # Demoireing dataset
        if Demoireing_path is not None:
            self.dataset_path_gt_Demoireing = os.path.join(Demoireing_path, 'Train_256/target')
            self.dataset_path_lq_Demoireing = os.path.join(Demoireing_path, 'Train_256/input')
            self.paths_gt_Demoireing, self.sizes_gt_Demoireing = util.get_image_paths('img', self.dataset_path_gt_Demoireing)
            self.paths_lq_Demoireing, self.sizes_lq_Demoireing = util.get_image_paths('img', self.dataset_path_lq_Demoireing)
            self.paths_Demoireing_len = len(self.paths_lq_Demoireing)
            sorted(self.paths_gt_Demoireing)
            sorted(self.paths_lq_Demoireing)

        # DustRemoval dataset
        if DustRemoval_path is not None:
            self.dataset_path_gt_DustRemoval = os.path.join(DustRemoval_path, 'Train_256/target')
            self.dataset_path_lq_DustRemoval = os.path.join(DustRemoval_path, 'Train_256/input')
            self.paths_gt_DustRemoval, self.sizes_gt_DustRemoval = util.get_image_paths('img', self.dataset_path_gt_DustRemoval)
            self.paths_lq_DustRemoval, self.sizes_lq_DustRemoval = util.get_image_paths('img', self.dataset_path_lq_DustRemoval)
            self.paths_DustRemoval_len = len(self.paths_lq_DustRemoval)
            sorted(self.paths_gt_DustRemoval)
            sorted(self.paths_lq_DustRemoval)
        
        # Desnow dataset
        if Desnow_path is not None:
            self.dataset_path_gt_Desnow = os.path.join(Desnow_path, 'Train_256/target')
            self.dataset_path_lq_Desnow = os.path.join(Desnow_path, 'Train_256/input')
            self.paths_gt_Desnow, self.sizes_gt_Desnow = util.get_image_paths('img', self.dataset_path_gt_Desnow)
            self.paths_lq_Desnow, self.sizes_lq_Desnow = util.get_image_paths('img', self.dataset_path_lq_Desnow)
            self.paths_Desnow_len = len(self.paths_lq_Desnow)
            sorted(self.paths_gt_Desnow)
            sorted(self.paths_lq_Desnow)

        # FlareRemoval dataset
        if FlareRemoval_path is not None:
            self.dataset_path_gt_FlareRemoval = os.path.join(FlareRemoval_path, 'Train_256/target')
            self.dataset_path_lq_FlareRemoval = os.path.join(FlareRemoval_path, 'Train_256/input')
            self.paths_gt_FlareRemoval, self.sizes_gt_FlareRemoval = util.get_image_paths('img', self.dataset_path_gt_FlareRemoval)
            self.paths_lq_FlareRemoval, self.sizes_lq_FlareRemoval = util.get_image_paths('img', self.dataset_path_lq_FlareRemoval)
            self.paths_FlareRemoval_len = len(self.paths_lq_FlareRemoval)
            sorted(self.paths_gt_FlareRemoval)
            sorted(self.paths_lq_FlareRemoval)

        # HighlightRemoval dataset
        if HighlightRemoval_path is not None:
            self.dataset_path_gt_HighlightRemoval = os.path.join(HighlightRemoval_path, 'Train_256/target')
            self.dataset_path_lq_HighlightRemoval = os.path.join(HighlightRemoval_path, 'Train_256/input')
            self.paths_gt_HighlightRemoval, self.sizes_gt_HighlightRemoval = util.get_image_paths('img', self.dataset_path_gt_HighlightRemoval)
            self.paths_lq_HighlightRemoval, self.sizes_lq_HighlightRemoval = util.get_image_paths('img', self.dataset_path_lq_HighlightRemoval)
            self.paths_HighlightRemoval_len = len(self.paths_lq_HighlightRemoval)
            sorted(self.paths_gt_HighlightRemoval)
            sorted(self.paths_lq_HighlightRemoval)


        ##########################################################################
        # Enhancement
        # LowLight dataset
        if LowLight_path is not None:
            self.dataset_path_gt_LowLight = os.path.join(LowLight_path, 'Train_256/target')
            self.dataset_path_lq_LowLight = os.path.join(LowLight_path, 'Train_256/input')
            self.paths_gt_LowLight, self.sizes_gt_LowLight = util.get_image_paths('img', self.dataset_path_gt_LowLight)
            self.paths_lq_LowLight, self.sizes_lq_LowLight = util.get_image_paths('img', self.dataset_path_lq_LowLight)
            self.paths_LowLight_len = len(self.paths_lq_LowLight)
            sorted(self.paths_gt_LowLight)
            sorted(self.paths_lq_LowLight)

        # BacklitEnhance dataset
        if BacklitEnhance_path is not None:
            self.dataset_path_gt_BacklitEnhance = os.path.join(BacklitEnhance_path, 'Train_256/target')
            self.dataset_path_lq_BacklitEnhance = os.path.join(BacklitEnhance_path, 'Train_256/input')
            self.paths_gt_BacklitEnhance, self.sizes_gt_BacklitEnhance = util.get_image_paths('img', self.dataset_path_gt_BacklitEnhance)
            self.paths_lq_BacklitEnhance, self.sizes_lq_BacklitEnhance = util.get_image_paths('img', self.dataset_path_lq_BacklitEnhance)
            self.paths_BacklitEnhance_len = len(self.paths_lq_BacklitEnhance)
            sorted(self.paths_gt_BacklitEnhance)
            sorted(self.paths_lq_BacklitEnhance)
        
        # ExpoCorrect dataset - Notice loader
        if ExpoCorrect_path is not None:
            self.dataset_path_gt_ExpoCorrect = os.path.join(ExpoCorrect_path, 'Train_256/target')
            self.dataset_path_lq_ExpoCorrect = os.path.join(ExpoCorrect_path, 'Train_256/input')
            self.paths_gt_ExpoCorrect, self.sizes_gt_ExpoCorrect = util.get_image_paths('img', self.dataset_path_gt_ExpoCorrect)
            self.paths_lq_ExpoCorrect, self.sizes_lq_ExpoCorrect = util.get_image_paths('img', self.dataset_path_lq_ExpoCorrect)
            self.paths_gt_ExpoCorrect_len = len(self.paths_gt_ExpoCorrect)
            self.paths_lq_ExpoCorrect_len = len(self.paths_lq_ExpoCorrect)
            sorted(self.paths_gt_ExpoCorrect)
            sorted(self.paths_lq_ExpoCorrect)

        # ISP dataset
        if ISP_path is not None:
            self.dataset_path_gt_ISP = os.path.join(ISP_path, 'Train_256/target')
            self.dataset_path_lq_ISP = os.path.join(ISP_path, 'Train_256/input')
            self.paths_gt_ISP, self.sizes_gt_ISP = util.get_image_paths('img', self.dataset_path_gt_ISP)
            self.paths_lq_ISP, self.sizes_lq_ISP = util.get_image_paths('img', self.dataset_path_lq_ISP)
            self.paths_ISP_len = len(self.paths_lq_ISP)
            sorted(self.paths_gt_ISP)
            sorted(self.paths_lq_ISP)

        # RenderBokeh dataset
        if RenderBokeh_path is not None:
            self.dataset_path_gt_RenderBokeh = os.path.join(RenderBokeh_path, 'Train_256/target')
            self.dataset_path_lq_RenderBokeh = os.path.join(RenderBokeh_path, 'Train_256/input')
            self.paths_gt_RenderBokeh, self.sizes_gt_RenderBokeh = util.get_image_paths('img', self.dataset_path_gt_RenderBokeh)
            self.paths_lq_RenderBokeh, self.sizes_lq_RenderBokeh = util.get_image_paths('img', self.dataset_path_lq_RenderBokeh)
            self.paths_RenderBokeh_len = len(self.paths_lq_RenderBokeh)
            sorted(self.paths_gt_RenderBokeh)
            sorted(self.paths_lq_RenderBokeh)

        # HistoEqual dataset
        if HistoEqual_path is not None:
            self.dataset_path_gt_HistoEqual = os.path.join(HistoEqual_path, 'Train_256/target')
            self.dataset_path_lq_HistoEqual = os.path.join(HistoEqual_path, 'Train_256/input')
            self.paths_gt_HistoEqual, self.sizes_gt_HistoEqual = util.get_image_paths('img', self.dataset_path_gt_HistoEqual)
            self.paths_lq_HistoEqual, self.sizes_lq_HistoEqual = util.get_image_paths('img', self.dataset_path_lq_HistoEqual)
            self.paths_HistoEqual_len = len(self.paths_lq_HistoEqual)
            sorted(self.paths_gt_HistoEqual)
            sorted(self.paths_lq_HistoEqual)

        # ColorCorrect dataset
        if ColorCorrect_path is not None:
            self.dataset_path_gt_ColorCorrect = os.path.join(ColorCorrect_path, 'Train_256/target')
            self.dataset_path_lq_ColorCorrect = os.path.join(ColorCorrect_path, 'Train_256/input')
            self.paths_gt_ColorCorrect, self.sizes_gt_ColorCorrect = util.get_image_paths('img', self.dataset_path_gt_ColorCorrect)
            self.paths_lq_ColorCorrect, self.sizes_lq_ColorCorrect = util.get_image_paths('img', self.dataset_path_lq_ColorCorrect)
            self.paths_ColorCorrect_len = len(self.paths_lq_ColorCorrect)
            sorted(self.paths_gt_ColorCorrect)
            sorted(self.paths_lq_ColorCorrect)

        # VignettingRemoval dataset
        if VignettingRemoval_path is not None:
            self.dataset_path_gt_VignettingRemoval = os.path.join(VignettingRemoval_path, 'Train_256/target')
            self.dataset_path_lq_VignettingRemoval = os.path.join(VignettingRemoval_path, 'Train_256/input')
            self.paths_gt_VignettingRemoval, self.sizes_gt_VignettingRemoval = util.get_image_paths('img', self.dataset_path_gt_VignettingRemoval)
            self.paths_lq_VignettingRemoval, self.sizes_lq_VignettingRemoval = util.get_image_paths('img', self.dataset_path_lq_VignettingRemoval)
            self.paths_VignettingRemoval_len = len(self.paths_lq_VignettingRemoval)
            sorted(self.paths_gt_VignettingRemoval)
            sorted(self.paths_lq_VignettingRemoval)

        # PhotoRetouch dataset
        if PhotoRetouch_path is not None:
            self.dataset_path_gt_PhotoRetouch = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.dataset_path_lq_PhotoRetouch = os.path.join(PhotoRetouch_path, 'Train_256/input')
            self.paths_gt_PhotoRetouch, self.sizes_gt_PhotoRetouch = util.get_image_paths('img', self.dataset_path_gt_PhotoRetouch)
            self.paths_lq_PhotoRetouch, self.sizes_lq_PhotoRetouch = util.get_image_paths('img', self.dataset_path_lq_PhotoRetouch)
            self.paths_PhotoRetouch_len = len(self.paths_lq_PhotoRetouch)
            sorted(self.paths_gt_PhotoRetouch)
            sorted(self.paths_lq_PhotoRetouch)

        # LocalLapFilter dataset - Notice loader
        if LocalLapFilter_path is not None:
            self.dataset_path_gt_LocalLapFilter = os.path.join(LocalLapFilter_path, 'Train_256_target')
            self.dataset_path_lq_LocalLapFilter = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_LocalLapFilter, self.sizes_gt_LocalLapFilter = util.get_image_paths('img', self.dataset_path_gt_LocalLapFilter)
            self.paths_lq_LocalLapFilter, self.sizes_lq_LocalLapFilter = util.get_image_paths('img', self.dataset_path_lq_LocalLapFilter)
            self.paths_LocalLapFilter_len = len(self.paths_lq_LocalLapFilter)
            sorted(self.paths_gt_LocalLapFilter)
            sorted(self.paths_lq_LocalLapFilter)

        # MultiScaleTM dataset
        if MultiScaleTM_path is not None:
            self.dataset_path_gt_MultiScaleTM = os.path.join(MultiScaleTM_path, 'Train_256_target')
            self.dataset_path_lq_MultiScaleTM = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_MultiScaleTM, self.sizes_gt_MultiScaleTM = util.get_image_paths('img', self.dataset_path_gt_MultiScaleTM)
            self.paths_lq_MultiScaleTM, self.sizes_lq_MultiScaleTM = util.get_image_paths('img', self.dataset_path_lq_MultiScaleTM)
            self.paths_MultiScaleTM_len = len(self.paths_lq_MultiScaleTM)
            sorted(self.paths_gt_MultiScaleTM)
            sorted(self.paths_lq_MultiScaleTM)

        # WhiteBalance dataset - Notice
        if WhiteBalance_path is not None:
            # Multiple lq images to one gt image
            self.dataset_path_gt_WhiteBalance = os.path.join(WhiteBalance_path, 'Train_256/target')
            self.dataset_path_lq_WhiteBalance = os.path.join(WhiteBalance_path, 'Train_256/input')
            self.paths_gt_WhiteBalance, self.sizes_gt_WhiteBalance = util.get_image_paths('img', self.dataset_path_gt_WhiteBalance)
            self.paths_lq_WhiteBalance, self.sizes_lq_WhiteBalance = util.get_image_paths('img', self.dataset_path_lq_WhiteBalance)
            self.paths_gt_WhiteBalance_len = len(self.paths_gt_WhiteBalance)
            self.paths_lq_WhiteBalance_len = len(self.paths_lq_WhiteBalance)
            # sorted(self.paths_gt_WhiteBalance)
            sorted(self.paths_lq_WhiteBalance)

        # SDH-HDR dataset - Notice
        if SDRHDR_path is not None:
            self.dataset_path_SDR = os.path.join(SDRHDR_path, 'SDR/Train_256')
            self.dataset_path_HDR = os.path.join(SDRHDR_path, 'HDR/Train_256')
            self.paths_SDR, self.sizes_SDR = util.get_image_paths('img', self.dataset_path_SDR)
            self.paths_HDR, self.sizes_HDR = util.get_image_paths('img', self.dataset_path_HDR)
            self.paths_SDR_len = len(self.paths_SDR)
            sorted(self.paths_SDR)
            sorted(self.paths_HDR)

        # InstagramFilter dataset - Notice
        if InstagramFilter_path is not None:
            # Different folder structure
            self.dataset_path_Ins_Ori = os.path.join(InstagramFilter_path, 'Train_256/original')
            self.dataset_path_Ins_Enh = os.path.join(InstagramFilter_path, 'Train_256/processed')
            self.paths_Ins_Ori, self.sizes_Ins_Ori = util.get_image_paths('img', self.dataset_path_Ins_Ori)
            self.paths_Ins_Enh, self.sizes_Ins_Enh = util.get_image_paths('img', self.dataset_path_Ins_Enh)
            self.Ori_paths_InstagramFilter_len = len(self.paths_Ins_Ori)
            self.Enh_paths_InstagramFilter_len = len(self.paths_Ins_Enh)
            sorted(self.paths_Ins_Ori)
            sorted(self.paths_Ins_Enh)


        ##########################################################################
        # FeatureExtra
        # DepthEstimate dataset
        if DepthEstimate_path is not None:
            self.dataset_path_gt_DepthEstimate = os.path.join(DepthEstimate_path, 'Train_256/target')
            self.dataset_path_lq_DepthEstimate = os.path.join(DepthEstimate_path, 'Train_256/input')
            self.paths_gt_DepthEstimate, self.sizes_gt_DepthEstimate = util.get_image_paths('img', self.dataset_path_gt_DepthEstimate)
            self.paths_lq_DepthEstimate, self.sizes_lq_DepthEstimate = util.get_image_paths('img', self.dataset_path_lq_DepthEstimate)
            self.paths_DepthEstimate_len = len(self.paths_lq_DepthEstimate)
            sorted(self.paths_gt_DepthEstimate)
            sorted(self.paths_lq_DepthEstimate)
        
        # PercepEdgeDetect dataset
        if PercepEdgeDetect_path is not None:
            self.dataset_path_gt_PercepEdgeDetect = os.path.join(PercepEdgeDetect_path, 'Train_256/target')
            self.dataset_path_lq_PercepEdgeDetect = os.path.join(PercepEdgeDetect_path, 'Train_256/input')
            self.paths_gt_PercepEdgeDetect, self.sizes_gt_PercepEdgeDetect = util.get_image_paths('img', self.dataset_path_gt_PercepEdgeDetect)
            self.paths_lq_PercepEdgeDetect, self.sizes_lq_PercepEdgeDetect = util.get_image_paths('img', self.dataset_path_lq_PercepEdgeDetect)
            self.paths_PercepEdgeDetect_len = len(self.paths_lq_PercepEdgeDetect)
            sorted(self.paths_gt_PercepEdgeDetect)
            sorted(self.paths_lq_PercepEdgeDetect)
        
        # SaliencyObject dataset
        if SaliencyObject_path is not None:
            self.dataset_path_gt_SaliencyObject = os.path.join(SaliencyObject_path, 'Train_256/target')
            self.dataset_path_lq_SaliencyObject = os.path.join(SaliencyObject_path, 'Train_256/input')
            self.paths_gt_SaliencyObject, self.sizes_gt_SaliencyObject = util.get_image_paths('img', self.dataset_path_gt_SaliencyObject)
            self.paths_lq_SaliencyObject, self.sizes_lq_SaliencyObject = util.get_image_paths('img', self.dataset_path_lq_SaliencyObject)
            self.paths_SaliencyObject_len = len(self.paths_lq_SaliencyObject)
            sorted(self.paths_gt_SaliencyObject)
            sorted(self.paths_lq_SaliencyObject)
        
        # HoughLine dataset
        if HoughLine_path is not None:
            self.dataset_path_gt_HoughLine = os.path.join(HoughLine_path, 'Train_256/target')
            self.dataset_path_lq_HoughLine = os.path.join(HoughLine_path, 'Train_256/input')
            self.paths_gt_HoughLine, self.sizes_gt_HoughLine = util.get_image_paths('img', self.dataset_path_gt_HoughLine)
            self.paths_lq_HoughLine, self.sizes_lq_HoughLine = util.get_image_paths('img', self.dataset_path_lq_HoughLine)
            self.paths_HoughLine_len = len(self.paths_lq_HoughLine)
            sorted(self.paths_gt_HoughLine)
            sorted(self.paths_lq_HoughLine)
        
        # Normal dataset
        if Normal_path is not None:
            self.dataset_path_gt_Normal = os.path.join(Normal_path, 'Train_256/target')
            self.dataset_path_lq_Normal = os.path.join(Normal_path, 'Train_256/input')
            self.paths_gt_Normal, self.sizes_gt_Normal = util.get_image_paths('img', self.dataset_path_gt_Normal)
            self.paths_lq_Normal, self.sizes_lq_Normal = util.get_image_paths('img', self.dataset_path_lq_Normal)
            self.paths_Normal_len = len(self.paths_lq_Normal)
            sorted(self.paths_gt_Normal)
            sorted(self.paths_lq_Normal)
        
        # HEDBoundary dataset
        if HEDBoundary_path is not None:
            self.dataset_path_gt_HEDBoundary = os.path.join(HEDBoundary_path, 'Train_256/target')
            self.dataset_path_lq_HEDBoundary = os.path.join(HEDBoundary_path, 'Train_256/input')
            self.paths_gt_HEDBoundary, self.sizes_gt_HEDBoundary = util.get_image_paths('img', self.dataset_path_gt_HEDBoundary)
            self.paths_lq_HEDBoundary, self.sizes_lq_HEDBoundary = util.get_image_paths('img', self.dataset_path_lq_HEDBoundary)
            self.paths_HEDBoundary_len = len(self.paths_lq_HEDBoundary)
            sorted(self.paths_gt_HEDBoundary)
            sorted(self.paths_lq_HEDBoundary)
        

        ##########################################################################
        # Stylization
        # PencilDrawing dataset - Notice loader
        if PencilDrawing_path is not None:
            self.dataset_path_gt_PencilDrawing = os.path.join(PencilDrawing_path, 'Train_256_target')
            self.dataset_path_lq_PencilDrawing = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_PencilDrawing, self.sizes_gt_PencilDrawing = util.get_image_paths('img', self.dataset_path_gt_PencilDrawing)
            self.paths_lq_PencilDrawing, self.sizes_lq_PencilDrawing = util.get_image_paths('img', self.dataset_path_lq_PencilDrawing)
            self.paths_PencilDrawing_len = len(self.paths_lq_PencilDrawing)
            sorted(self.paths_gt_PencilDrawing)
            sorted(self.paths_lq_PencilDrawing)
        
        # Photographic dataset - Notice loader
        if Photographic_path is not None:
            self.dataset_path_gt_Photographic = os.path.join(Photographic_path, 'Train_256_target')
            self.dataset_path_lq_Photographic = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Photographic, self.sizes_gt_Photographic = util.get_image_paths('img', self.dataset_path_gt_Photographic)
            self.paths_lq_Photographic, self.sizes_lq_Photographic = util.get_image_paths('img', self.dataset_path_lq_Photographic)
            self.paths_Photographic_len = len(self.paths_lq_Photographic)
            sorted(self.paths_gt_Photographic)
            sorted(self.paths_lq_Photographic)
            
        # RTV dataset - Notice loader
        if RTV_path is not None:
            self.dataset_path_gt_RTV = os.path.join(RTV_path, 'Train_256_target')
            self.dataset_path_lq_RTV = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_RTV, self.sizes_gt_RTV = util.get_image_paths('img', self.dataset_path_gt_RTV)
            self.paths_lq_RTV, self.sizes_lq_RTV = util.get_image_paths('img', self.dataset_path_lq_RTV)
            self.paths_RTV_len = len(self.paths_lq_RTV)
            sorted(self.paths_gt_RTV)
            sorted(self.paths_lq_RTV)
        
        # Cloisonnism dataset - Notice loader
        if Cloisonnism_path is not None:
            self.dataset_path_gt_Cloisonnism = os.path.join(Cloisonnism_path, 'Train_256_target')
            self.dataset_path_lq_Cloisonnism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Cloisonnism, self.sizes_gt_Cloisonnism = util.get_image_paths('img', self.dataset_path_gt_Cloisonnism)
            
            style1 = [p for p in self.paths_gt_Cloisonnism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Cloisonnism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Cloisonnism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Cloisonnism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Cloisonnism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Cloisonnism_singlelist_len = len(style1)
            self.lists_Cloisonnism = [style1, style2, style3, style4, style5]
        
        # Divisionism dataset - Notice loader
        if Divisionism_path is not None:
            self.dataset_path_gt_Divisionism = os.path.join(Divisionism_path, 'Train_256_target')
            self.dataset_path_lq_Divisionism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Divisionism, self.sizes_gt_Divisionism = util.get_image_paths('img', self.dataset_path_gt_Divisionism)
            
            style1 = [p for p in self.paths_gt_Divisionism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Divisionism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Divisionism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Divisionism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Divisionism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Divisionism_singlelist_len = len(style1)
            self.lists_Divisionism = [style1, style2, style3, style4, style5]
        
        # Fauvism dataset - Notice loader
        if Fauvism_path is not None:
            self.dataset_path_gt_Fauvism = os.path.join(Fauvism_path, 'Train_256_target')
            self.dataset_path_lq_Fauvism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Fauvism, self.sizes_gt_Fauvism = util.get_image_paths('img', self.dataset_path_gt_Fauvism)
            
            style1 = [p for p in self.paths_gt_Fauvism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Fauvism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Fauvism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Fauvism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Fauvism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Fauvism_singlelist_len = len(style1)
            self.lists_Fauvism = [style1, style2, style3, style4, style5]
        
        # Vermeer dataset - Notice loader
        if Vermeer_path is not None:
            self.dataset_path_gt_Vermeer = os.path.join(Vermeer_path, 'Train_256_target')
            self.dataset_path_lq_Vermeer = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Vermeer, self.sizes_gt_Vermeer = util.get_image_paths('img', self.dataset_path_gt_Vermeer)
            
            style1 = [p for p in self.paths_gt_Vermeer if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Vermeer if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Vermeer if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Vermeer if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Vermeer if os.path.basename(p)[:2]=='5_']
            
            self.paths_Vermeer_singlelist_len = len(style1)
            self.lists_Vermeer = [style1, style2, style3, style4, style5]
        
        # JOJO dataset - Notice loader
        if JOJO_path is not None:
            self.dataset_path_gt_JOJO = os.path.join(JOJO_path, 'Train_256_target')
            self.dataset_path_lq_JOJO = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_JOJO, self.sizes_gt_JOJO = util.get_image_paths('img', self.dataset_path_gt_JOJO)
            
            style1 = [p for p in self.paths_gt_JOJO if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_JOJO if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_JOJO if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_JOJO if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_JOJO if os.path.basename(p)[:2]=='5_']
            
            self.paths_JOJO_singlelist_len = len(style1)
            self.lists_JOJO = [style1, style2, style3, style4, style5]
        
        # Raphael dataset - Notice loader
        if Raphael_path is not None:
            self.dataset_path_gt_Raphael = os.path.join(Raphael_path, 'Train_256_target')
            self.dataset_path_lq_Raphael = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Raphael, self.sizes_gt_Raphael = util.get_image_paths('img', self.dataset_path_gt_Raphael)
            
            style1 = [p for p in self.paths_gt_Raphael if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Raphael if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Raphael if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Raphael if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Raphael if os.path.basename(p)[:2]=='5_']
            
            self.paths_Raphael_singlelist_len = len(style1)
            self.lists_Raphael = [style1, style2, style3, style4, style5]
        
        # Modernism dataset - Notice loader
        if Modernism_path is not None:
            self.dataset_path_gt_Modernism = os.path.join(Modernism_path, 'Train_256_target')
            self.dataset_path_lq_Modernism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Modernism, self.sizes_gt_Modernism = util.get_image_paths('img', self.dataset_path_gt_Modernism)
            
            style1 = [p for p in self.paths_gt_Modernism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Modernism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Modernism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Modernism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Modernism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Modernism_singlelist_len = len(style1)
            self.lists_Modernism = [style1, style2, style3, style4, style5]
        
        # Monet dataset - Notice loader
        if Monet_path is not None:
            self.dataset_path_gt_Monet = os.path.join(Monet_path, 'Train_256_target')
            self.dataset_path_lq_Monet = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Monet, self.sizes_gt_Monet = util.get_image_paths('img', self.dataset_path_gt_Monet)
            
            style1 = [p for p in self.paths_gt_Monet if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Monet if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Monet if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Monet if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Monet if os.path.basename(p)[:2]=='5_']
            
            self.paths_Monet_singlelist_len = len(style1)
            self.lists_Monet = [style1, style2, style3, style4, style5]

        # NeoImpressionism dataset - Notice loader
        if NeoImpressionism_path is not None:
            self.dataset_path_gt_NeoImpressionism = os.path.join(NeoImpressionism_path, 'Train_256_target')
            self.dataset_path_lq_NeoImpressionism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_NeoImpressionism, self.sizes_gt_NeoImpressionism = util.get_image_paths('img', self.dataset_path_gt_NeoImpressionism)
            
            style1 = [p for p in self.paths_gt_NeoImpressionism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_NeoImpressionism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_NeoImpressionism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_NeoImpressionism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_NeoImpressionism if os.path.basename(p)[:2]=='5_']
            
            self.paths_NeoImpressionism_singlelist_len = len(style1)
            self.lists_NeoImpressionism = [style1, style2, style3, style4, style5]
        
        # PopArt dataset - Notice loader
        if PopArt_path is not None:
            self.dataset_path_gt_PopArt = os.path.join(PopArt_path, 'Train_256_target')
            self.dataset_path_lq_PopArt = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_PopArt, self.sizes_gt_PopArt = util.get_image_paths('img', self.dataset_path_gt_PopArt)
            
            style1 = [p for p in self.paths_gt_PopArt if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_PopArt if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_PopArt if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_PopArt if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_PopArt if os.path.basename(p)[:2]=='5_']
            
            self.paths_PopArt_singlelist_len = len(style1)
            self.lists_PopArt = [style1, style2, style3, style4, style5]
        
        # Ukiyoe dataset - Notice loader
        if Ukiyoe_path is not None:
            self.dataset_path_gt_Ukiyoe = os.path.join(Ukiyoe_path, 'Train_256_target')
            self.dataset_path_lq_Ukiyoe = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Ukiyoe, self.sizes_gt_Ukiyoe = util.get_image_paths('img', self.dataset_path_gt_Ukiyoe)
            
            style1 = [p for p in self.paths_gt_Ukiyoe if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Ukiyoe if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Ukiyoe if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Ukiyoe if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Ukiyoe if os.path.basename(p)[:2]=='5_']
            
            self.paths_Ukiyoe_singlelist_len = len(style1)
            self.lists_Ukiyoe = [style1, style2, style3, style4, style5]
        
        # VanGogh dataset - Notice loader
        if VanGogh_path is not None:
            self.dataset_path_gt_VanGogh = os.path.join(VanGogh_path, 'Train_256_target')
            self.dataset_path_lq_VanGogh = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_VanGogh, self.sizes_gt_VanGogh = util.get_image_paths('img', self.dataset_path_gt_VanGogh)
            
            style1 = [p for p in self.paths_gt_VanGogh if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_VanGogh if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_VanGogh if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_VanGogh if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_VanGogh if os.path.basename(p)[:2]=='5_']
            
            self.paths_VanGogh_singlelist_len = len(style1)
            self.lists_VanGogh = [style1, style2, style3, style4, style5]
        
        # Tuner dataset - Notice loader
        if Tuner_path is not None:
            self.dataset_path_gt_Tuner = os.path.join(Tuner_path, 'Train_256_target')
            self.dataset_path_lq_Tuner = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Tuner, self.sizes_gt_Tuner = util.get_image_paths('img', self.dataset_path_gt_Tuner)
            
            style1 = [p for p in self.paths_gt_Tuner if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Tuner if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Tuner if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Tuner if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Tuner if os.path.basename(p)[:2]=='5_']
            
            self.paths_Tuner_singlelist_len = len(style1)
            self.lists_Tuner = [style1, style2, style3, style4, style5]
        
        # Regionalism dataset - Notice loader
        if Regionalism_path is not None:
            self.dataset_path_gt_Regionalism = os.path.join(Regionalism_path, 'Train_256_target')
            self.dataset_path_lq_Regionalism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Regionalism, self.sizes_gt_Regionalism = util.get_image_paths('img', self.dataset_path_gt_Regionalism)
            
            style1 = [p for p in self.paths_gt_Regionalism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Regionalism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Regionalism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Regionalism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Regionalism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Regionalism_singlelist_len = len(style1)
            self.lists_Regionalism = [style1, style2, style3, style4, style5]
        
        # Impressionism dataset - Notice loader
        if Impressionism_path is not None:
            self.dataset_path_gt_Impressionism = os.path.join(Impressionism_path, 'Train_256_target')
            self.dataset_path_lq_Impressionism = os.path.join(PhotoRetouch_path, 'Train_256/target')
            self.paths_gt_Impressionism, self.sizes_gt_Impressionism = util.get_image_paths('img', self.dataset_path_gt_Impressionism)
            
            style1 = [p for p in self.paths_gt_Impressionism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Impressionism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Impressionism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Impressionism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Impressionism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Impressionism_singlelist_len = len(style1)
            self.lists_Impressionism = [style1, style2, style3, style4, style5]


        ##########################################################################
        # Special Data
        # Face dataset - Notice loader
        if Face_path is not None:
            self.dataset_path_gt_Face = os.path.join(Face_path, 'Train_target_256')
            self.dataset_path_lr_Face = os.path.join(Face_path, 'Train_X4Down_256')
            self.paths_gt_Face, self.sizes_gt_Face = util.get_image_paths('img', self.dataset_path_gt_Face)
            self.paths_lr_Face, self.sizes_lr_Face = util.get_image_paths('img', self.dataset_path_lr_Face)
            self.paths_Face_len = len(self.paths_gt_Face)
            sorted(self.paths_gt_Face)
            sorted(self.paths_lr_Face)
        
        # Infrared dataset - Notice loader
        if Infrared_path is not None:
            self.dataset_path_gt_Infrared = os.path.join(Infrared_path, 'Train_target_256')
            self.dataset_path_lr_Infrared = os.path.join(Infrared_path, 'Train_X4Down_256')
            self.paths_gt_Infrared, self.sizes_gt_Infrared = util.get_image_paths('img', self.dataset_path_gt_Infrared)
            self.paths_lr_Infrared, self.sizes_lr_Infrared = util.get_image_paths('img', self.dataset_path_lr_Infrared)
            self.paths_Infrared_len = len(self.paths_gt_Infrared)
            sorted(self.paths_gt_Infrared)
            sorted(self.paths_lr_Infrared)

        # CT dataset - Notice loader
        if CT_path is not None:
            self.dataset_path_gt_CT = os.path.join(CT_path, 'Train_target_256')
            self.dataset_path_lr_CT = os.path.join(CT_path, 'Train_X4Down_256')
            self.dataset_path_ns_CT = os.path.join(CT_path, 'Train_GN15_256')
            self.paths_gt_CT, self.sizes_gt_CT = util.get_image_paths('img', self.dataset_path_gt_CT)
            self.paths_lr_CT, self.sizes_lr_CT = util.get_image_paths('img', self.dataset_path_lr_CT)
            self.paths_ns_CT, self.sizes_ns_CT = util.get_image_paths('img', self.dataset_path_ns_CT)
            self.paths_CT_len = len(self.paths_gt_CT)
            sorted(self.paths_gt_CT)
            sorted(self.paths_lr_CT)
            sorted(self.paths_ns_CT)
        
        # MRI dataset - Notice loader
        if MRI_path is not None:
            self.dataset_path_gt_MRI = os.path.join(MRI_path, 'Train_target_256')
            self.dataset_path_lr_MRI = os.path.join(MRI_path, 'Train_X4Down_256')
            self.dataset_path_ns_MRI = os.path.join(MRI_path, 'Train_GN15_256')
            self.paths_gt_MRI, self.sizes_gt_MRI = util.get_image_paths('img', self.dataset_path_gt_MRI)
            self.paths_lr_MRI, self.sizes_lr_MRI = util.get_image_paths('img', self.dataset_path_lr_MRI)
            self.paths_ns_MRI, self.sizes_ns_MRI = util.get_image_paths('img', self.dataset_path_ns_MRI)
            self.paths_MRI_len = len(self.paths_gt_MRI)
            sorted(self.paths_gt_MRI)
            sorted(self.paths_lr_MRI)
            sorted(self.paths_ns_MRI)
        
        # Satellite dataset - Notice loader
        if Satellite_path is not None:
            self.dataset_path_gt_Satellite = os.path.join(Satellite_path, 'Train_target_256')
            self.dataset_path_lr_Satellite = os.path.join(Satellite_path, 'Train_X4Down_256')
            self.dataset_path_ns_Satellite = os.path.join(Satellite_path, 'Train_GN15_256')
            self.paths_gt_Satellite, self.sizes_gt_Satellite = util.get_image_paths('img', self.dataset_path_gt_Satellite)
            self.paths_lr_Satellite, self.sizes_lr_Satellite = util.get_image_paths('img', self.dataset_path_lr_Satellite)
            self.paths_ns_Satellite, self.sizes_ns_Satellite = util.get_image_paths('img', self.dataset_path_ns_Satellite)
            self.paths_Satellite_len = len(self.paths_gt_Satellite)
            sorted(self.paths_gt_Satellite)
            sorted(self.paths_lr_Satellite)
            sorted(self.paths_ns_Satellite)
        
         # Weather dataset - Notice loader
        if Weather_path is not None:
            self.dataset_path_gt_Weather = os.path.join(Weather_path, 'Train_target_256')
            self.dataset_path_lr_Weather = os.path.join(Weather_path, 'Train_X4Down_256')
            self.paths_gt_Weather, self.sizes_gt_Weather = util.get_image_paths('img', self.dataset_path_gt_Weather)
            self.paths_lr_Weather, self.sizes_lr_Weather = util.get_image_paths('img', self.dataset_path_lr_Weather)
            self.paths_Weather_len = len(self.paths_gt_Weather)
            sorted(self.paths_gt_Weather)
            sorted(self.paths_lr_Weather)
        
        # FluidFlow dataset - Notice loader
        if FluidFlow_path is not None:
            self.dataset_path_gt_FluidFlow = os.path.join(FluidFlow_path, 'Train_target_256')
            self.dataset_path_lr_FluidFlow = os.path.join(FluidFlow_path, 'Train_X4Down_256')
            self.paths_gt_FluidFlow, self.sizes_gt_FluidFlow = util.get_image_paths('img', self.dataset_path_gt_FluidFlow)
            self.paths_lr_FluidFlow, self.sizes_lr_FluidFlow = util.get_image_paths('img', self.dataset_path_lr_FluidFlow)
            self.paths_FluidFlow_len = len(self.paths_gt_FluidFlow)
            sorted(self.paths_gt_FluidFlow)
            sorted(self.paths_lr_FluidFlow)
        
        # Cosmology dataset - Notice loader
        if Cosmology_path is not None:
            self.dataset_path_gt_Cosmology = os.path.join(Cosmology_path, 'Train_target_256')
            self.dataset_path_lr_Cosmology = os.path.join(Cosmology_path, 'Train_X4Down_256')
            self.paths_gt_Cosmology, self.sizes_gt_Cosmology = util.get_image_paths('img', self.dataset_path_gt_Cosmology)
            self.paths_lr_Cosmology, self.sizes_lr_Cosmology = util.get_image_paths('img', self.dataset_path_lr_Cosmology)
            self.paths_Cosmology_len = len(self.paths_gt_Cosmology)
            sorted(self.paths_gt_Cosmology)
            sorted(self.paths_lr_Cosmology)


        ##########################################################################
        # dataset list   
        if self.tasks_version == 0:
            self.dataset_list = ['Base', 'Derain', 'RainDrop', 'MarineSnowRemoval', 'ReflectionRemoval', 'ShadowRemovalISTD', # 1-6
                                'ShadowRemovalSRD', 'CloudRemoval', 'WatermarkRemoval', 'RealLLSR', 'UDCPoled', 'UDCToled', # 7-12
                                'Dehaze', 'Demoireing', 'DustRemoval', 'Desnow', 'FlareRemoval', 'HighlightRemoval', 'ISP', # 13-19
                                'LowLight', 'BacklitEnhance', 'ExpoCorrect', 'RenderBokeh', 'HistoEqual', 'ColorCorrect', # 20-25
                                'VignettingRemoval', 'PhotoRetouch', 'LocalLapFilter', 'MultiScaleTM', 'WhiteBalance', # 26-30
                                'SDRHDR', 'InstagramFilter', 'DepthEstimate', 'PercepEdgeDetect', 'SaliencyObject', # 31-35
                                'HoughLine', 'Normal', 'HEDBoundary', 'PencilDrawing', 'Photographic', 'RTV', 'Cloisonnism', # 36-42
                                'Divisionism', 'Fauvism', 'Vermeer', 'JOJO', 'Raphael', 'Modernism', 'Monet', 'PopArt', # 43-50
                                'NeoImpressionism', 'Ukiyoe', 'VanGogh', 'Tuner', 'Regionalism', 'Impressionism', 'Face', # 51-57
                                'Infrared', 'CT', 'MRI', 'Satellite', 'Weather', 'FluidFlow', 'Cosmology'] # 58-64
            # single task
            self.onthefly_degradation_list1 = ['blur', 'noise', 'compression', 'brighten', 'darken', 'spatter',
                                               'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen',
                                               'saturate_weaken', 'oversharpen', 'pixelate', 'quantization']
            self.onthefly_degradation_list2 = ['Rain', 'Ringing', 'r_l', 'Inpainting', 'mosaic', 'SRx2', 'SRx4']
            self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        else:
            print('Wrong task flag!')
            sys.exit(1)
            
        print('dataset list: ', self.dataset_list)
        print('degradation_type list: ', self.onthefly_degradation_list1 + self.onthefly_degradation_list2 + self.onthefly_degradation_list3)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return self.paths_base_len

    def __getitem__(self, idx):
        if self.tasks_version == 0:
            dataset_choice = np.random.choice(self.dataset_list, p=[33/100, 1/100, 1/100, 1/100, 1/100, 0.5/100, 0.5/100, 1/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 2/100, 2/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100,
                                                                    1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100,
                                                                    1/100, 1/100, 2/100, 2/100, 2/100, 1/100, 1/100, 1/100,
                                                                    ])
        else:
            print("Wrong tasks_version!")
            sys.exit(1)

        if dataset_choice == 'Base':
            random_index1 = random.randint(0, self.paths_base_len-1)
            gt1_path = self.paths_base_gt[random_index1]
            random_index2 = random.randint(0, self.paths_base_len-1)
            gt2_path = self.paths_base_gt[random_index2]
            
            img_gt1 = util.read_img(None, gt1_path, None, float=False) # np.uint8
            img_gt2 = util.read_img(None, gt2_path, None, float=False) # np.uint8
            
            # 3-channel images
            if img_gt1.ndim == 2:
                img_gt1 = np.expand_dims(img_gt1, axis=2)
                img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
            if img_gt2.ndim == 2:
                img_gt2 = np.expand_dims(img_gt2, axis=2)
                img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)

            if img_gt1.shape[2] !=3:
                img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
            if img_gt2.shape[2] !=3:
                img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)

            # if the image size is not same as self.gt_size
            H, W, _ = img_gt1.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size), interpolation=cv2.INTER_CUBIC)
            else: 
                img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)

            H, W, _ = img_gt2.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size), interpolation=cv2.INTER_CUBIC)
            else: 
                img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            
            # add on-the-fly degradation
            tasks_list = ['X-Distortion', 'Extra-Degradation', 'Operators', 'Mixture']
            tasks_flag = np.random.choice(tasks_list, p=[20/33, 7/33, 2/33, 4/33])

            if tasks_flag == 'X-Distortion':
                deg_type = random.choice(self.onthefly_degradation_list1)
                img_lq1, img_lq2, img_gt1, img_gt2, deg_type = add_x_distortion_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
                img_lq1 = uint2single(img_lq1)
                img_lq2 = uint2single(img_lq2)
                img_gt1 = uint2single(img_gt1)
                img_gt2 = uint2single(img_gt2)
            
            elif tasks_flag == 'Operators':
                deg_type = random.choice(self.onthefly_degradation_list3)
                img_lq1, img_lq2, img_gt1, img_gt2 = calculate_operators_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
                img_lq1 = uint2single(img_lq1)
                img_lq2 = uint2single(img_lq2)
                img_gt1 = uint2single(img_gt1)
                img_gt2 = uint2single(img_gt2)

            elif tasks_flag == 'Extra-Degradation':
                img_gt1 = uint2single(img_gt1)
                img_gt2 = uint2single(img_gt2)
                deg_type = random.choice(self.onthefly_degradation_list2)
                img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)

            elif tasks_flag == 'Mixture':
                # 1 round Mix-degradation
                degradation_type_1 = ['Rain', 'spatter', 'blur', 'None', 'None']
                degradation_type_2 = ['brighten', 'darken', 'saturate_strengthen', 'saturate_weaken', 'None']
                degradation_type_3 = ['noise', 'None', 'None', 'None', 'None']   
                degradation_type_4 = ['Ringing', 'r_l', 'None', 'None', 'None'] 
                degradation_type_5 = ['compression', 'None', 'None', 'None', 'None']

                deg_type_list = [random.choice(degradation_type_1), 
                                 random.choice(degradation_type_2), 
                                 random.choice(degradation_type_3), 
                                 random.choice(degradation_type_4),
                                 random.choice(degradation_type_5)]
                
                img_lq1 = np.copy(img_gt1) # uint8
                img_lq2 = np.copy(img_gt2) # uint8
                for deg_type in deg_type_list:
                    if deg_type == 'None':
                        continue
                    elif deg_type in self.onthefly_degradation_list1:
                        img_lq1, img_lq2, _, _, _ = add_x_distortion_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
                    else:
                        img_lq1 = uint2single(img_lq1)
                        img_lq2 = uint2single(img_lq2)
                        img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type)
                        img_lq1 = single2uint(img_lq1)
                        img_lq2 = single2uint(img_lq2)
                img_lq1 = uint2single(img_lq1)
                img_lq2 = uint2single(img_lq2)
                img_gt1 = uint2single(img_gt1)
                img_gt2 = uint2single(img_gt2)
                deg_type = 'Mix1R_' + '-'.join(deg_type_list)
                    

        # Restoration
        # Derain task
        elif dataset_choice == 'Derain':
            random_index1 = random.randint(0, self.paths_Derain_len-1)
            lq1_path = self.paths_lq_Derain[random_index1]
            gt1_path = self.paths_gt_Derain[random_index1]
    
            random_index2 = random.randint(0, self.paths_Derain_len-1)
            lq2_path = self.paths_lq_Derain[random_index2]
            gt2_path = self.paths_gt_Derain[random_index2]
            
            deg_type = 'Derain'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # RainDrop task
        elif dataset_choice == 'RainDrop':
            random_index1 = random.randint(0, self.paths_RainDrop_len-1)
            lq1_path = self.paths_lq_RainDrop[random_index1]
            gt1_path = self.paths_gt_RainDrop[random_index1]
    
            random_index2 = random.randint(0, self.paths_RainDrop_len-1)
            lq2_path = self.paths_lq_RainDrop[random_index2]
            gt2_path = self.paths_gt_RainDrop[random_index2]
            
            deg_type = 'RainDrop'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # MarineSnowRemoval task
        elif dataset_choice == 'MarineSnowRemoval':
            random_index1 = random.randint(0, self.paths_MarineSnowRemoval_len-1)
            lq1_path = self.paths_lq_MarineSnowRemoval[random_index1]
            gt1_path = self.paths_gt_MarineSnowRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_MarineSnowRemoval_len-1)
            lq2_path = self.paths_lq_MarineSnowRemoval[random_index2]
            gt2_path = self.paths_gt_MarineSnowRemoval[random_index2]
            
            deg_type = 'MarineSnowRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # ReflectionRemoval task
        elif dataset_choice == 'ReflectionRemoval':
            random_index1 = random.randint(0, self.paths_ReflectionRemoval_len-1)
            lq1_path = self.paths_lq_ReflectionRemoval[random_index1]
            gt1_path = self.paths_gt_ReflectionRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_ReflectionRemoval_len-1)
            lq2_path = self.paths_lq_ReflectionRemoval[random_index2]
            gt2_path = self.paths_gt_ReflectionRemoval[random_index2]
            
            deg_type = 'ReflectionRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # ShadowRemovalISTD task
        elif dataset_choice == 'ShadowRemovalISTD':
            random_index1 = random.randint(0, self.paths_ShadowRemovalISTD_len-1)
            lq1_path = self.paths_lq_ShadowRemovalISTD[random_index1]
            gt1_path = self.paths_gt_ShadowRemovalISTD[random_index1]
    
            random_index2 = random.randint(0, self.paths_ShadowRemovalISTD_len-1)
            lq2_path = self.paths_lq_ShadowRemovalISTD[random_index2]
            gt2_path = self.paths_gt_ShadowRemovalISTD[random_index2]
            
            deg_type = 'ShadowRemovalISTD'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # ShadowRemovalSRD task
        elif dataset_choice == 'ShadowRemovalSRD':
            random_index1 = random.randint(0, self.paths_ShadowRemovalSRD_len-1)
            lq1_path = self.paths_lq_ShadowRemovalSRD[random_index1]
            gt1_path = self.paths_gt_ShadowRemovalSRD[random_index1]
    
            random_index2 = random.randint(0, self.paths_ShadowRemovalSRD_len-1)
            lq2_path = self.paths_lq_ShadowRemovalSRD[random_index2]
            gt2_path = self.paths_gt_ShadowRemovalSRD[random_index2]
            
            deg_type = 'ShadowRemovalSRD'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # CloudRemoval task
        elif dataset_choice == 'CloudRemoval':
            random_index1 = random.randint(0, self.paths_CloudRemoval_len-1)
            lq1_path = self.paths_lq_CloudRemoval[random_index1]
            gt1_path = self.paths_gt_CloudRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_CloudRemoval_len-1)
            lq2_path = self.paths_lq_CloudRemoval[random_index2]
            gt2_path = self.paths_gt_CloudRemoval[random_index2]
            
            deg_type = 'CloudRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # WatermarkRemoval task
        elif dataset_choice == 'WatermarkRemoval':
            random_index1 = random.randint(0, self.paths_WatermarkRemoval_len-1)
            lq1_path = self.paths_lq_WatermarkRemoval[random_index1]
            gt1_path = self.paths_gt_WatermarkRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_WatermarkRemoval_len-1)
            lq2_path = self.paths_lq_WatermarkRemoval[random_index2]
            gt2_path = self.paths_gt_WatermarkRemoval[random_index2]
            
            deg_type = 'WatermarkRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # RealLLSR task - Notice
        elif dataset_choice == 'RealLLSR':
            random_index1 = random.randint(0, self.paths_lq_RealLLSR_len-1)
            lq1_path = self.paths_lq_RealLLSR[random_index1]
            gt1_name = lq1_path.split('/')[-1].split('-')[0]
            expo_suffix = lq1_path.split('-')[-1]
            gt1_path = os.path.join(self.dataset_path_gt_RealLLSR, '{}.png'.format(gt1_name))
            
            random_index2 = random.randint(0, self.paths_gt_RealLLSR_len-1)
            gt2_path = self.paths_gt_RealLLSR[random_index2]
            lq2_name = '-'.join([os.path.splitext(gt2_path.split('/')[-1])[0], expo_suffix])
            lq2_path = os.path.join(self.dataset_path_lq_RealLLSR, lq2_name)

            if not os.path.exists(lq2_path):
                expo = expo_suffix[:-4]
                if expo == '2.0':
                    lq2_path = lq2_path.replace(expo, '2.5')
                elif expo == '2.5':
                    lq2_path = lq2_path.replace(expo, '3.0')
                elif expo == '4.5':
                    lq2_path = lq2_path.replace(expo, '4.0')
                elif expo == '5.0':
                    lq2_path = lq2_path.replace(expo, '4.5')
            
            if not os.path.exists(lq2_path):
                expo = lq2_path.split('-')[-1][:-4]
                if expo == '2.5':
                    lq2_path = lq2_path.replace(expo, '3.0')
                elif expo == '4.5':
                    lq2_path = lq2_path.replace(expo, '4.0')

            deg_type = 'RealLowLightSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # UDCPoled task
        elif dataset_choice == 'UDCPoled':
            random_index1 = random.randint(0, self.paths_UDCPoled_len-1)
            lq1_path = self.paths_lq_UDCPoled[random_index1]
            gt1_path = self.paths_gt_UDCPoled[random_index1]
    
            random_index2 = random.randint(0, self.paths_UDCPoled_len-1)
            lq2_path = self.paths_lq_UDCPoled[random_index2]
            gt2_path = self.paths_gt_UDCPoled[random_index2]
            
            deg_type = 'UDCPoled'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # UDCToled task
        elif dataset_choice == 'UDCToled':
            random_index1 = random.randint(0, self.paths_UDCToled_len-1)
            lq1_path = self.paths_lq_UDCToled[random_index1]
            gt1_path = self.paths_gt_UDCToled[random_index1]
    
            random_index2 = random.randint(0, self.paths_UDCToled_len-1)
            lq2_path = self.paths_lq_UDCToled[random_index2]
            gt2_path = self.paths_gt_UDCToled[random_index2]
            
            deg_type = 'UDCToled'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Dehaze task - Notice
        elif dataset_choice == 'Dehaze':
            random_index1 = random.randint(0, self.paths_Dehaze_len-1)
            lq1_path = self.paths_lq_Dehaze[random_index1]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            gt1_path = os.path.join(self.dataset_path_gt_Dehaze, '{}.png'.format(gt1_name))
    
            random_index2 = random.randint(0, self.paths_Dehaze_len-1)
            lq2_path = self.paths_lq_Dehaze[random_index2]
            gt2_name = lq2_path.split('/')[-1].split('_')[0]
            gt2_path = os.path.join(self.dataset_path_gt_Dehaze, '{}.png'.format(gt2_name))

            deg_type = 'Dehaze'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Demoireing task
        elif dataset_choice == 'Demoireing':
            random_index1 = random.randint(0, self.paths_Demoireing_len-1)
            lq1_path = self.paths_lq_Demoireing[random_index1]
            gt1_path = self.paths_gt_Demoireing[random_index1]
    
            random_index2 = random.randint(0, self.paths_Demoireing_len-1)
            lq2_path = self.paths_lq_Demoireing[random_index2]
            gt2_path = self.paths_gt_Demoireing[random_index2]
            
            deg_type = 'Demoireing'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # DustRemoval task
        elif dataset_choice == 'DustRemoval':
            random_index1 = random.randint(0, self.paths_DustRemoval_len-1)
            lq1_path = self.paths_lq_DustRemoval[random_index1]
            gt1_path = self.paths_gt_DustRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_DustRemoval_len-1)
            lq2_path = self.paths_lq_DustRemoval[random_index2]
            gt2_path = self.paths_gt_DustRemoval[random_index2]
            
            deg_type = 'DustRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Desnow task
        elif dataset_choice == 'Desnow':
            random_index1 = random.randint(0, self.paths_Desnow_len-1)
            lq1_path = self.paths_lq_Desnow[random_index1]
            gt1_path = self.paths_gt_Desnow[random_index1]
    
            random_index2 = random.randint(0, self.paths_Desnow_len-1)
            lq2_path = self.paths_lq_Desnow[random_index2]
            gt2_path = self.paths_gt_Desnow[random_index2]
            
            deg_type = 'Desnow'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # FlareRemoval task
        elif dataset_choice == 'FlareRemoval':
            random_index1 = random.randint(0, self.paths_FlareRemoval_len-1)
            lq1_path = self.paths_lq_FlareRemoval[random_index1]
            gt1_path = self.paths_gt_FlareRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_FlareRemoval_len-1)
            lq2_path = self.paths_lq_FlareRemoval[random_index2]
            gt2_path = self.paths_gt_FlareRemoval[random_index2]
            
            deg_type = 'FlareRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # HighlightRemoval task
        elif dataset_choice == 'HighlightRemoval':
            random_index1 = random.randint(0, self.paths_HighlightRemoval_len-1)
            lq1_path = self.paths_lq_HighlightRemoval[random_index1]
            gt1_path = self.paths_gt_HighlightRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_HighlightRemoval_len-1)
            lq2_path = self.paths_lq_HighlightRemoval[random_index2]
            gt2_path = self.paths_gt_HighlightRemoval[random_index2]
            
            deg_type = 'HighlightRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)


        # Enhancement
        # LowLight task
        elif dataset_choice == 'LowLight':
            random_index1 = random.randint(0, self.paths_LowLight_len-1)
            lq1_path = self.paths_lq_LowLight[random_index1]
            gt1_path = self.paths_gt_LowLight[random_index1]
    
            random_index2 = random.randint(0, self.paths_LowLight_len-1)
            lq2_path = self.paths_lq_LowLight[random_index2]
            gt2_path = self.paths_gt_LowLight[random_index2]
            
            deg_type = 'LowLight'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # BacklitEnhance task
        elif dataset_choice == 'BacklitEnhance':
            random_index1 = random.randint(0, self.paths_BacklitEnhance_len-1)
            lq1_path = self.paths_lq_BacklitEnhance[random_index1]
            gt1_path = self.paths_gt_BacklitEnhance[random_index1]
    
            random_index2 = random.randint(0, self.paths_BacklitEnhance_len-1)
            lq2_path = self.paths_lq_BacklitEnhance[random_index2]
            gt2_path = self.paths_gt_BacklitEnhance[random_index2]
            
            deg_type = 'BacklitEnhance'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # ExpoCorrect task  - Notice
        elif dataset_choice == 'ExpoCorrect':
            # multiple lq to one gt
            random_index1 = random.randint(0, self.paths_lq_ExpoCorrect_len-1)
            lq1_path = self.paths_lq_ExpoCorrect[random_index1]
            gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-1])
            expo_suffix = lq1_path.split('_')[-1]
            gt1_path = os.path.join(self.dataset_path_gt_ExpoCorrect, '{}.jpg'.format(gt1_name))
            
            random_index2 = random.randint(0, self.paths_gt_ExpoCorrect_len-1)
            gt2_path = self.paths_gt_ExpoCorrect[random_index2]
            lq2_name = '_'.join([os.path.splitext(gt2_path.split('/')[-1])[0], expo_suffix])
            lq2_path = os.path.join(self.dataset_path_lq_ExpoCorrect, lq2_name)
            
            deg_type = 'ExpoCorrect'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # ISP task
        elif dataset_choice == 'ISP':
            random_index1 = random.randint(0, self.paths_ISP_len-1)
            lq1_path = self.paths_lq_ISP[random_index1]
            gt1_path = self.paths_gt_ISP[random_index1]
    
            random_index2 = random.randint(0, self.paths_ISP_len-1)
            lq2_path = self.paths_lq_ISP[random_index2]
            gt2_path = self.paths_gt_ISP[random_index2]
            
            deg_type = 'ISP'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # RenderBokeh task
        elif dataset_choice == 'RenderBokeh':
            random_index1 = random.randint(0, self.paths_RenderBokeh_len-1)
            lq1_path = self.paths_lq_RenderBokeh[random_index1]
            gt1_path = self.paths_gt_RenderBokeh[random_index1]
    
            random_index2 = random.randint(0, self.paths_RenderBokeh_len-1)
            lq2_path = self.paths_lq_RenderBokeh[random_index2]
            gt2_path = self.paths_gt_RenderBokeh[random_index2]
            
            deg_type = 'RenderBokeh'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # HistoEqual task
        elif dataset_choice == 'HistoEqual':
            random_index1 = random.randint(0, self.paths_HistoEqual_len-1)
            lq1_path = self.paths_lq_HistoEqual[random_index1]
            gt1_path = self.paths_gt_HistoEqual[random_index1]
    
            random_index2 = random.randint(0, self.paths_HistoEqual_len-1)
            lq2_path = self.paths_lq_HistoEqual[random_index2]
            gt2_path = self.paths_gt_HistoEqual[random_index2]
            
            deg_type = 'HistoEqual'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # ColorCorrect task
        elif dataset_choice == 'ColorCorrect':
            random_index1 = random.randint(0, self.paths_ColorCorrect_len-1)
            lq1_path = self.paths_lq_ColorCorrect[random_index1]
            gt1_path = self.paths_gt_ColorCorrect[random_index1]
    
            random_index2 = random.randint(0, self.paths_ColorCorrect_len-1)
            lq2_path = self.paths_lq_ColorCorrect[random_index2]
            gt2_path = self.paths_gt_ColorCorrect[random_index2]
            
            deg_type = 'ColorCorrect'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # VignettingRemoval task
        elif dataset_choice == 'VignettingRemoval':
            random_index1 = random.randint(0, self.paths_VignettingRemoval_len-1)
            lq1_path = self.paths_lq_VignettingRemoval[random_index1]
            gt1_path = self.paths_gt_VignettingRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_VignettingRemoval_len-1)
            lq2_path = self.paths_lq_VignettingRemoval[random_index2]
            gt2_path = self.paths_gt_VignettingRemoval[random_index2]
            
            deg_type = 'VignettingRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # PhotoRetouch task
        elif dataset_choice == 'PhotoRetouch':
            random_index1 = random.randint(0, self.paths_PhotoRetouch_len-1)
            lq1_path = self.paths_lq_PhotoRetouch[random_index1]
            gt1_path = self.paths_gt_PhotoRetouch[random_index1]
    
            random_index2 = random.randint(0, self.paths_PhotoRetouch_len-1)
            lq2_path = self.paths_lq_PhotoRetouch[random_index2]
            gt2_path = self.paths_gt_PhotoRetouch[random_index2]
            
            deg_type = 'PhotoRetouch'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # LocalLapFilter task
        elif dataset_choice == 'LocalLapFilter':
            random_index1 = random.randint(0, self.paths_LocalLapFilter_len-1)
            lq1_path = self.paths_lq_LocalLapFilter[random_index1]
            gt1_path = self.paths_gt_LocalLapFilter[random_index1]
            
            random_index2 = random.randint(0, self.paths_LocalLapFilter_len-1)
            lq2_path = self.paths_lq_LocalLapFilter[random_index2]
            gt2_path = self.paths_gt_LocalLapFilter[random_index2]
            
            deg_type = 'LocalLapFilter'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # MultiScaleTM task
        elif dataset_choice == 'MultiScaleTM':
            random_index1 = random.randint(0, self.paths_MultiScaleTM_len-1)
            lq1_path = self.paths_lq_MultiScaleTM[random_index1]
            gt1_path = self.paths_gt_MultiScaleTM[random_index1]
            
            random_index2 = random.randint(0, self.paths_MultiScaleTM_len-1)
            lq2_path = self.paths_lq_MultiScaleTM[random_index2]
            gt2_path = self.paths_gt_MultiScaleTM[random_index2]
            
            deg_type = 'MultiScaleTM'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # WhiteBalance task  - Notice
        elif dataset_choice == 'WhiteBalance':
            # multiple lq to one gt
            # random_index1 = random.randint(0, self.paths_lq_WhiteBalance_len-1)
            # lq1_path = self.paths_lq_WhiteBalance[random_index1]
            # gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-2])
            # wb_suffix = '_'.join(lq1_path.split('_')[-2:])
            # gt1_path = os.path.join(self.dataset_path_gt_WhiteBalance, '{}_G_AS.png'.format(gt1_name))
            
            # random_index2 = random.randint(0, self.paths_gt_WhiteBalance_len-1)
            # gt2_path = self.paths_gt_WhiteBalance[random_index2]
            # lq2_name = gt2_path.split('/')[-1].replace('G_AS.png', wb_suffix)
            # lq2_path = os.path.join(self.dataset_path_lq_WhiteBalance, lq2_name)

            random_index1 = random.randint(0, self.paths_lq_WhiteBalance_len-1)
            lq1_path = self.paths_lq_WhiteBalance[random_index1]
            gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-2])
            gt1_path = os.path.join(self.dataset_path_gt_WhiteBalance, '{}_G_AS.png'.format(gt1_name))

            random_index2 = random.randint(0, self.paths_lq_WhiteBalance_len-1)
            lq2_path = self.paths_lq_WhiteBalance[random_index2]
            gt2_name = '_'.join(lq2_path.split('/')[-1].split('_')[:-2])
            gt2_path = os.path.join(self.dataset_path_gt_WhiteBalance, '{}_G_AS.png'.format(gt2_name))

            deg_type = 'WhiteBalance'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # SDR-to-HDR & HDR-to-SDR task - Notice
        elif dataset_choice == 'SDRHDR':
            random_index1 = random.randint(0, self.paths_SDR_len-1)
            sdr1_path = self.paths_SDR[random_index1]
            hdr1_path = self.paths_HDR[random_index1]
            
            random_index2 = random.randint(0, self.paths_SDR_len-1)
            sdr2_path = self.paths_SDR[random_index2]
            hdr2_path = self.paths_HDR[random_index2]
            
            task_select = np.random.choice(['SDR2HDR', 'HDR2SDR'])
            if task_select == 'SDR2HDR':
                deg_type = 'SDR2HDR'
                img_gt1 = util.read_img(None, hdr1_path, None)
                img_lq1 = util.read_img(None, sdr1_path, None) 
                img_gt2 = util.read_img(None, hdr2_path, None)
                img_lq2 = util.read_img(None, sdr2_path, None)
            else:
                deg_type = 'HDR2SDR'
                img_gt1 = util.read_img(None, sdr1_path, None)
                img_lq1 = util.read_img(None, hdr1_path, None) 
                img_gt2 = util.read_img(None, sdr2_path, None)
                img_lq2 = util.read_img(None, hdr2_path, None)

        # InsFilterRemoval & InsFilterAddition task - Notice
        elif dataset_choice == 'InstagramFilter':
            random_index1 = random.randint(0, self.Enh_paths_InstagramFilter_len-1)
            enh1_path = self.paths_Ins_Enh[random_index1]
            filter_suffix = enh1_path.split('/')[-1].split('_')[-1]
            ori1_name = enh1_path.split('/')[-1].split('_')[0]
            ori1_path = os.path.join(self.dataset_path_Ins_Ori, '{}_Original.jpg'.format(ori1_name))
            
            random_index2 = random.randint(0, self.Ori_paths_InstagramFilter_len-1)
            ori2_path = self.paths_Ins_Ori[random_index2]
            enh2_name = '_'.join([ori2_path.split('/')[-1].split('_')[0], filter_suffix])
            ehn2_path = os.path.join(self.dataset_path_Ins_Enh, enh2_name)
            
            task_select = np.random.choice(['InsFilterRemoval', 'InsFilterAddition'])
            if task_select == 'InsFilterRemoval':
                deg_type = 'InsFilterRemoval'
                img_gt1 = util.read_img(None, ori1_path, None)
                img_lq1 = util.read_img(None, enh1_path, None) 
                img_gt2 = util.read_img(None, ori2_path, None)
                img_lq2 = util.read_img(None, ehn2_path, None)
            else:
                deg_type = 'InsFilterAddition'
                img_gt1 = util.read_img(None, enh1_path, None)
                img_lq1 = util.read_img(None, ori1_path, None) 
                img_gt2 = util.read_img(None, ehn2_path, None)
                img_lq2 = util.read_img(None, ori2_path, None)


        # FeatureExtra
        # DepthEstimate task
        elif dataset_choice == 'DepthEstimate':
            random_index1 = random.randint(0, self.paths_DepthEstimate_len-1)
            lq1_path = self.paths_lq_DepthEstimate[random_index1]
            gt1_path = self.paths_gt_DepthEstimate[random_index1]
    
            random_index2 = random.randint(0, self.paths_DepthEstimate_len-1)
            lq2_path = self.paths_lq_DepthEstimate[random_index2]
            gt2_path = self.paths_gt_DepthEstimate[random_index2]
            
            deg_type = 'DepthEstimate'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # PercepEdgeDetect task
        elif dataset_choice == 'PercepEdgeDetect':
            random_index1 = random.randint(0, self.paths_PercepEdgeDetect_len-1)
            lq1_path = self.paths_lq_PercepEdgeDetect[random_index1]
            gt1_path = self.paths_gt_PercepEdgeDetect[random_index1]
    
            random_index2 = random.randint(0, self.paths_PercepEdgeDetect_len-1)
            lq2_path = self.paths_lq_PercepEdgeDetect[random_index2]
            gt2_path = self.paths_gt_PercepEdgeDetect[random_index2]
            
            deg_type = 'PercepEdgeDetect'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # SaliencyObject task
        elif dataset_choice == 'SaliencyObject':
            random_index1 = random.randint(0, self.paths_SaliencyObject_len-1)
            lq1_path = self.paths_lq_SaliencyObject[random_index1]
            gt1_path = self.paths_gt_SaliencyObject[random_index1]
    
            random_index2 = random.randint(0, self.paths_SaliencyObject_len-1)
            lq2_path = self.paths_lq_SaliencyObject[random_index2]
            gt2_path = self.paths_gt_SaliencyObject[random_index2]
            
            deg_type = 'SaliencyObject'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # HoughLine task
        elif dataset_choice == 'HoughLine':
            random_index1 = random.randint(0, self.paths_HoughLine_len-1)
            lq1_path = self.paths_lq_HoughLine[random_index1]
            gt1_path = self.paths_gt_HoughLine[random_index1]
    
            random_index2 = random.randint(0, self.paths_HoughLine_len-1)
            lq2_path = self.paths_lq_HoughLine[random_index2]
            gt2_path = self.paths_gt_HoughLine[random_index2]
            
            deg_type = 'HoughLine'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Normal task
        elif dataset_choice == 'Normal':
            random_index1 = random.randint(0, self.paths_Normal_len-1)
            lq1_path = self.paths_lq_Normal[random_index1]
            gt1_path = self.paths_gt_Normal[random_index1]
    
            random_index2 = random.randint(0, self.paths_Normal_len-1)
            lq2_path = self.paths_lq_Normal[random_index2]
            gt2_path = self.paths_gt_Normal[random_index2]
            
            deg_type = 'Normal'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # HEDBoundary task
        elif dataset_choice == 'HEDBoundary':
            random_index1 = random.randint(0, self.paths_HEDBoundary_len-1)
            lq1_path = self.paths_lq_HEDBoundary[random_index1]
            gt1_path = self.paths_gt_HEDBoundary[random_index1]
    
            random_index2 = random.randint(0, self.paths_HEDBoundary_len-1)
            lq2_path = self.paths_lq_HEDBoundary[random_index2]
            gt2_path = self.paths_gt_HEDBoundary[random_index2]
            
            deg_type = 'HEDBoundary'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)


        # FeatureExtra
        # PencilDrawing task
        elif dataset_choice == 'PencilDrawing':
            random_index1 = random.randint(0, self.paths_PencilDrawing_len-1)
            lq1_path = self.paths_lq_PencilDrawing[random_index1]
            gt1_path = self.paths_gt_PencilDrawing[random_index1]
    
            random_index2 = random.randint(0, self.paths_PencilDrawing_len-1)
            lq2_path = self.paths_lq_PencilDrawing[random_index2]
            gt2_path = self.paths_gt_PencilDrawing[random_index2]
            
            deg_type = 'PencilDrawing'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Photographic task
        elif dataset_choice == 'Photographic':
            random_index1 = random.randint(0, self.paths_Photographic_len-1)
            lq1_path = self.paths_lq_Photographic[random_index1]
            gt1_path = self.paths_gt_Photographic[random_index1]
    
            random_index2 = random.randint(0, self.paths_Photographic_len-1)
            lq2_path = self.paths_lq_Photographic[random_index2]
            gt2_path = self.paths_gt_Photographic[random_index2]
            
            deg_type = 'Photographic'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # RTV task
        elif dataset_choice == 'RTV':
            random_index1 = random.randint(0, self.paths_RTV_len-1)
            lq1_path = self.paths_lq_RTV[random_index1]
            gt1_path = self.paths_gt_RTV[random_index1]
    
            random_index2 = random.randint(0, self.paths_RTV_len-1)
            lq2_path = self.paths_lq_RTV[random_index2]
            gt2_path = self.paths_gt_RTV[random_index2]
            
            deg_type = 'RTV'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Cloisonnism task    
        elif dataset_choice == 'Cloisonnism':
            random_style_list = random.choice(self.lists_Cloisonnism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Cloisonnism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Cloisonnism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Cloisonnism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Divisionism task    
        elif dataset_choice == 'Divisionism':
            random_style_list = random.choice(self.lists_Divisionism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Divisionism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Divisionism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Divisionism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Fauvism task    
        elif dataset_choice == 'Fauvism':
            random_style_list = random.choice(self.lists_Fauvism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Fauvism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Fauvism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Fauvism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Vermeer task    
        elif dataset_choice == 'Vermeer':
            random_style_list = random.choice(self.lists_Vermeer)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Vermeer + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Vermeer + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Vermeer'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # JOJO task    
        elif dataset_choice == 'JOJO':
            random_style_list = random.choice(self.lists_JOJO)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_JOJO + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_JOJO + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'JOJO'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Raphael task    
        elif dataset_choice == 'Raphael':
            random_style_list = random.choice(self.lists_Raphael)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Raphael + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Raphael + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Raphael'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Modernism task    
        elif dataset_choice == 'Modernism':
            random_style_list = random.choice(self.lists_Modernism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Modernism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Modernism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Modernism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Monet task    
        elif dataset_choice == 'Monet':
            random_style_list = random.choice(self.lists_Monet)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Monet + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Monet + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Monet'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # NeoImpressionism task    
        elif dataset_choice == 'NeoImpressionism':
            random_style_list = random.choice(self.lists_NeoImpressionism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_NeoImpressionism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_NeoImpressionism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'NeoImpressionism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # PopArt task    
        elif dataset_choice == 'PopArt':
            random_style_list = random.choice(self.lists_PopArt)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_PopArt + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_PopArt + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'PopArt'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Ukiyoe task    
        elif dataset_choice == 'Ukiyoe':
            random_style_list = random.choice(self.lists_Ukiyoe)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Ukiyoe + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Ukiyoe + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Ukiyoe'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # VanGogh task    
        elif dataset_choice == 'VanGogh':
            random_style_list = random.choice(self.lists_VanGogh)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_VanGogh + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_VanGogh + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'VanGogh'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # Tuner task    
        elif dataset_choice == 'Tuner':
            random_style_list = random.choice(self.lists_Tuner)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Tuner + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Tuner + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Tuner'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Regionalism task    
        elif dataset_choice == 'Regionalism':
            random_style_list = random.choice(self.lists_Regionalism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Regionalism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Regionalism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Regionalism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Impressionism task    
        elif dataset_choice == 'Impressionism':
            random_style_list = random.choice(self.lists_Impressionism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Impressionism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Impressionism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'Impressionism'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # Special Data
        # FaceSR task
        elif dataset_choice == 'Face':
            random_index1 = random.randint(0, self.paths_Face_len-1)
            lq1_path = self.paths_lr_Face[random_index1]
            gt1_path = self.paths_gt_Face[random_index1]
    
            random_index2 = random.randint(0, self.paths_Face_len-1)
            lq2_path = self.paths_lr_Face[random_index2]
            gt2_path = self.paths_gt_Face[random_index2]
            
            deg_type = 'FaceSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # InfraredSR task
        elif dataset_choice == 'Infrared':
            random_index1 = random.randint(0, self.paths_Infrared_len-1)
            lq1_path = self.paths_lr_Infrared[random_index1]
            gt1_path = self.paths_gt_Infrared[random_index1]
    
            random_index2 = random.randint(0, self.paths_Infrared_len-1)
            lq2_path = self.paths_lr_Infrared[random_index2]
            gt2_path = self.paths_gt_Infrared[random_index2]
            
            deg_type = 'InfraredSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # CT SR & Denoise task
        elif dataset_choice == 'CT':
            random_index1 = random.randint(0, self.paths_CT_len-1)
            gt1_path = self.paths_gt_CT[random_index1]
    
            random_index2 = random.randint(0, self.paths_CT_len-1)
            gt2_path = self.paths_gt_CT[random_index2]
            
            task_select = np.random.choice(['SR', 'Denoise'])
            if task_select == 'SR':
                lq1_path = self.paths_lr_CT[random_index1]
                lq2_path = self.paths_lr_CT[random_index2]

                deg_type = 'CTSR'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)
            else:
                lq1_path = self.paths_ns_CT[random_index1]
                lq2_path = self.paths_ns_CT[random_index2]

                deg_type = 'CTDenoise'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)

        # MRI SR & Denoise task
        elif dataset_choice == 'MRI':
            random_index1 = random.randint(0, self.paths_MRI_len-1)
            gt1_path = self.paths_gt_MRI[random_index1]
    
            random_index2 = random.randint(0, self.paths_MRI_len-1)
            gt2_path = self.paths_gt_MRI[random_index2]
            
            task_select = np.random.choice(['SR', 'Denoise'])
            if task_select == 'SR':
                lq1_path = self.paths_lr_MRI[random_index1]
                lq2_path = self.paths_lr_MRI[random_index2]

                deg_type = 'MRISR'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)
            else:
                lq1_path = self.paths_ns_MRI[random_index1]
                lq2_path = self.paths_ns_MRI[random_index2]

                deg_type = 'MRIDenoise'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)
        
        # Satellite SR & Denoise task
        elif dataset_choice == 'Satellite':
            random_index1 = random.randint(0, self.paths_Satellite_len-1)
            gt1_path = self.paths_gt_Satellite[random_index1]
    
            random_index2 = random.randint(0, self.paths_Satellite_len-1)
            gt2_path = self.paths_gt_Satellite[random_index2]
            
            task_select = np.random.choice(['SR', 'Denoise'])
            if task_select == 'SR':
                lq1_path = self.paths_lr_Satellite[random_index1]
                lq2_path = self.paths_lr_Satellite[random_index2]

                deg_type = 'SatelliteSR'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)
            else:
                lq1_path = self.paths_ns_Satellite[random_index1]
                lq2_path = self.paths_ns_Satellite[random_index2]

                deg_type = 'SatelliteDenoise'
                img_gt1 = util.read_img(None, gt1_path, None)
                img_lq1 = util.read_img(None, lq1_path, None) 
                img_gt2 = util.read_img(None, gt2_path, None)
                img_lq2 = util.read_img(None, lq2_path, None)

        # WeatherSR task
        elif dataset_choice == 'Weather':
            random_index1 = random.randint(0, self.paths_Weather_len-1)
            lq1_path = self.paths_lr_Weather[random_index1]
            gt1_path = self.paths_gt_Weather[random_index1]
    
            random_index2 = random.randint(0, self.paths_Weather_len-1)
            lq2_path = self.paths_lr_Weather[random_index2]
            gt2_path = self.paths_gt_Weather[random_index2]
            
            deg_type = 'WeatherSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # FluidFlowSR task
        elif dataset_choice == 'FluidFlow':
            random_index1 = random.randint(0, self.paths_FluidFlow_len-1)
            lq1_path = self.paths_lr_FluidFlow[random_index1]
            gt1_path = self.paths_gt_FluidFlow[random_index1]
    
            random_index2 = random.randint(0, self.paths_FluidFlow_len-1)
            lq2_path = self.paths_lr_FluidFlow[random_index2]
            gt2_path = self.paths_gt_FluidFlow[random_index2]
            
            deg_type = 'FluidFlowSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        # CosmologySR task
        elif dataset_choice == 'Cosmology':
            random_index1 = random.randint(0, self.paths_Cosmology_len-1)
            lq1_path = self.paths_lr_Cosmology[random_index1]
            gt1_path = self.paths_gt_Cosmology[random_index1]
    
            random_index2 = random.randint(0, self.paths_Cosmology_len-1)
            lq2_path = self.paths_lr_Cosmology[random_index2]
            gt2_path = self.paths_gt_Cosmology[random_index2]
            
            deg_type = 'CosmologySR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()

        if img_gt1.ndim == 2:
            img_gt1 = np.expand_dims(img_gt1, axis=2)
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_gt2.ndim == 2:
            img_gt2 = np.expand_dims(img_gt2, axis=2)
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
            
        if img_gt1.shape[2] !=3:
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_gt2.shape[2] !=3:
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
            
        # augmentation - flip, rotate
        # img_lq1, img_lq2, img_gt1, img_gt2 = util.augment([img_lq1, img_lq2, img_gt1, img_gt2], hflip=True, rot=True)
                
        if img_lq1.shape != img_gt1.shape or img_lq2.shape != img_gt2.shape:
            print(deg_type)
            
        if not np.all(np.isfinite(np.concatenate((img_lq1, img_lq2, img_gt1, img_gt2), axis=2))):
            print("Data exists unfinite value.")
            sys.exit(1)

        # numpy to tensor
        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = torch.stack([img_lq1, img_gt1, img_lq2, img_gt2], dim=0)
        return batch, deg_type
    

class DatasetPrompt_Val(Dataset):
    def __init__(self, dataset_path, input_size=256, data_len=None):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.gt_size = input_size
        self.data_len = data_len

        random.seed(1000)
        if self.data_len is not None:
            random.shuffle(self.paths_gt)
        
        random.seed(2000)
        self.prompt_list = self.paths_gt.copy()
        random.shuffle(self.prompt_list)
        
        # on-the-fly single task
        self.onthefly_degradation_list1 = ['blur', 'noise', 'compression', 'brighten', 'darken', 'spatter',
                                            'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen',
                                            'saturate_weaken', 'oversharpen', 'pixelate', 'quantization']
        self.onthefly_degradation_list2 = ['Rain', 'Ringing', 'r_l', 'Inpainting', 'mosaic', 'SRx2', 'SRx4']
        self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        gt1_path = self.prompt_list[idx]
        gt2_path = self.paths_gt[idx]
        
        img_gt1 = util.read_img(None, gt1_path, None, float=False) # np.uint8
        img_gt2 = util.read_img(None, gt2_path, None, float=False) # np.uint8
        
        # 3-channel images
        if img_gt1.ndim == 2:
            img_gt1 = np.expand_dims(img_gt1, axis=2)
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_gt2.ndim == 2:
            img_gt2 = np.expand_dims(img_gt2, axis=2)
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)

        if img_gt1.shape[2] !=3:
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_gt2.shape[2] !=3:
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)

        # if the image size is not same as self.gt_size
        H, W, _ = img_gt1.shape
        if H < self.gt_size or W < self.gt_size:
            img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size), interpolation=cv2.INTER_CUBIC)
        else: 
            img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)

        H, W, _ = img_gt2.shape
        if H < self.gt_size or W < self.gt_size:
            img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size), interpolation=cv2.INTER_CUBIC)
        else: 
            img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            
        # add on-the-fly degradation
        tasks_list = ['X-Distortion', 'Extra-Degradation', 'Operators', 'Mixture']
        tasks_flag = np.random.choice(tasks_list, p=[20/33, 7/33, 2/33, 4/33])

        if tasks_flag == 'X-Distortion':
            deg_type = random.choice(self.onthefly_degradation_list1)
            img_lq1, img_lq2, img_gt1, img_gt2, deg_type = add_x_distortion_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
            img_lq1 = uint2single(img_lq1)
            img_lq2 = uint2single(img_lq2)
            img_gt1 = uint2single(img_gt1)
            img_gt2 = uint2single(img_gt2)
        
        elif tasks_flag == 'Operators':
            deg_type = random.choice(self.onthefly_degradation_list3)
            img_lq1, img_lq2, img_gt1, img_gt2 = calculate_operators_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
            img_lq1 = uint2single(img_lq1)
            img_lq2 = uint2single(img_lq2)
            img_gt1 = uint2single(img_gt1)
            img_gt2 = uint2single(img_gt2)

        elif tasks_flag == 'Extra-Degradation':
            img_gt1 = uint2single(img_gt1)
            img_gt2 = uint2single(img_gt2)
            deg_type = random.choice(self.onthefly_degradation_list2)
            img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)

        elif tasks_flag == 'Mixture':
            # 1 round Mix-degradation
            degradation_type_1 = ['Rain', 'spatter', 'blur', 'None', 'None']
            degradation_type_2 = ['brighten', 'darken', 'saturate_strengthen', 'saturate_weaken', 'None']
            degradation_type_3 = ['noise', 'None', 'None', 'None', 'None']   
            degradation_type_4 = ['Ringing', 'r_l', 'None', 'None', 'None'] 
            degradation_type_5 = ['compression', 'None', 'None', 'None', 'None']

            deg_type_list = [random.choice(degradation_type_1), 
                                random.choice(degradation_type_2), 
                                random.choice(degradation_type_3), 
                                random.choice(degradation_type_4),
                                random.choice(degradation_type_5)]
            
            img_lq1 = np.copy(img_gt1) # uint8
            img_lq2 = np.copy(img_gt2) # uint8
            for deg_type in deg_type_list:
                if deg_type == 'None':
                    continue
                elif deg_type in self.onthefly_degradation_list1:
                    img_lq1, img_lq2, _, _, _ = add_x_distortion_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
                else:
                    img_lq1 = uint2single(img_lq1)
                    img_lq2 = uint2single(img_lq2)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type)
                    img_lq1 = single2uint(img_lq1)
                    img_lq2 = single2uint(img_lq2)
            img_lq1 = uint2single(img_lq1)
            img_lq2 = uint2single(img_lq2)
            img_gt1 = uint2single(img_gt1)
            img_gt2 = uint2single(img_gt2)
            deg_type = 'Mix1R_' + '-'.join(deg_type_list)
                
        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
                 'input_query_img2': img_lq2, 'target_img2': img_gt2}
        return batch, deg_type

# class DatasetPrompt_Customized_Val(Dataset):
#     """
#     Dataset for customized data, random prompt 

#     Args:
#         Dataset (_type_): _description_
#     """
#     def __init__(self, dataset_path_gt, dataset_path_lq, input_size=256, dataset_type='SOTS', data_len=None):
#         self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path_gt)
#         self.paths_lq, self.sizes_lq = util.get_image_paths('img', dataset_path_lq)
#         self.dataset_path_gt = dataset_path_gt
#         self.dataset_path_lq = dataset_path_lq
#         self.input_size = input_size
#         sorted(self.paths_gt)
#         sorted(self.paths_lq)
#         self.dataset_type = dataset_type
#         self.data_len = data_len
        
#         random.seed(1000)
#         if self.data_len is not None:
#             pair_paths = list(zip(self.paths_gt, self.paths_lq))
#             random.shuffle(pair_paths)
#             prompt_list = pair_paths.copy()
#             random.shuffle(prompt_list)
#             self.paths_gt, self.paths_lq = zip(*pair_paths)
#             self.prompts_gt, self.prompts_lq = zip(*prompt_list)
        
#     def __len__(self):
#         if self.data_len is not None:
#             return self.data_len
#         else: return len(self.paths_lq)

#     def __getitem__(self, idx):
        
#         if self.dataset_type == 'SOTS':
#             lq1_path = self.prompts_lq[idx]
#             gt1_name = lq1_path.split('/')[-1].split('_')[0]
#             gt1_path = os.path.join(self.dataset_path_gt, '{}.png'.format(gt1_name))
            
#             lq2_path = self.paths_lq[idx]
#             gt2_name = lq2_path.split('/')[-1].split('_')[0]
#             gt2_path = os.path.join(self.dataset_path_gt, '{}.png'.format(gt2_name))
            
#             img_gt1 = util.read_img(None, gt1_path, None)
#             img_lq1 = util.read_img(None, lq1_path, None) 
#             img_gt2 = util.read_img(None, gt2_path, None)
#             img_lq2 = util.read_img(None, lq2_path, None)
            
#             H_gt, W_gt, _ = img_gt1.shape
#             H_lq, W_lq, _ = img_lq1.shape
            
#             crop_size_H = np.abs(H_lq-H_gt)//2
#             crop_size_W = np.abs(W_lq-W_gt)//2
#             img_gt1 = img_gt1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
#             img_gt2 = img_gt2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        
#         elif self.dataset_type == 'InsFilterRemoval':
#             lq1_path = self.prompts_lq[idx]
#             gt1_name = lq1_path.split('/')[-1].split('_')[0]
#             filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
#             gt1_path = os.path.join(self.dataset_path_gt, '{}_Original.jpg'.format(gt1_name))
            
#             gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
#             lq2_name = gt2_path.split('/')[-1].split('_')[0]
#             lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, filter_suffix))
            
#             img_gt1 = util.read_img(None, gt1_path, None)
#             img_lq1 = util.read_img(None, lq1_path, None) 
#             img_gt2 = util.read_img(None, gt2_path, None)
#             img_lq2 = util.read_img(None, lq2_path, None)
        
#         elif self.dataset_type == 'InsFilterAddition':
#             lq1_path = self.prompts_lq[idx]
#             gt1_name = lq1_path.split('/')[-1].split('_')[0]
#             filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
#             gt1_path = os.path.join(self.dataset_path_gt, '{}_Original.jpg'.format(gt1_name))
            
#             gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
#             lq2_name = gt2_path.split('/')[-1].split('_')[0]
#             lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, filter_suffix))
            
#             img_gt1 = util.read_img(None, lq1_path, None)
#             img_lq1 = util.read_img(None, gt1_path, None) 
#             img_gt2 = util.read_img(None, lq2_path, None)
#             img_lq2 = util.read_img(None, gt2_path, None)
        
#         elif self.dataset_type == 'ExposureError':
#             lq1_path = self.prompts_lq[idx]
#             gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-1])
#             expo_suffix = lq1_path.split('_')[-1]
#             gt1_path = os.path.join(self.dataset_path_gt, '{}.jpg'.format(gt1_name))
            
#             gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
#             lq2_name = os.path.splitext(gt2_path.split('/')[-1])[0]
#             lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, expo_suffix))
            
#             img_gt1 = util.read_img(None, gt1_path, None)
#             img_lq1 = util.read_img(None, lq1_path, None) 
#             img_gt2 = util.read_img(None, gt2_path, None)
#             img_lq2 = util.read_img(None, lq2_path, None)
        
#         # validation set is not prepared.
#         # elif self.dataset_type == 'WhiteBalance':
#         #     lq1_path = self.prompts_lq[idx]
#         #     gt1_name = lq1_path.split('/')[-1].split('_')[0]
#         #     gt1_path = os.path.join(self.dataset_path_gt, '{}_G_AS.png'.format(gt1_name))
            
#         #     lq2_path = self.paths_lq[idx]
#         #     gt2_name = lq2_path.split('/')[-1].split('_')[0]
#         #     gt2_path = os.path.join(self.dataset_path_gt, '{}_G_AS.png'.format(gt2_name))
            
#         #     img_gt1 = util.read_img(None, gt1_path, None)
#         #     img_lq1 = util.read_img(None, lq1_path, None) 
#         #     img_gt2 = util.read_img(None, gt2_path, None)
#         #     img_lq2 = util.read_img(None, lq2_path, None)
            
#         else:
#             gt1_path = self.prompts_gt[idx]
#             lq1_path = self.prompts_lq[idx]
            
#             gt2_path = self.paths_gt[idx]
#             lq2_path = self.paths_lq[idx]
#             img_gt1 = util.read_img(None, gt1_path, None)
#             img_lq1 = util.read_img(None, lq1_path, None) 
#             img_gt2 = util.read_img(None, gt2_path, None)
#             img_lq2 = util.read_img(None, lq2_path, None)
        
#         deg_type = self.dataset_type
        
#         # resize to fixed size
#         H, W, _ = img_lq1.shape
#         if H != self.input_size or W != self.input_size:
#             img_lq1 = cv2.resize(img_lq1, (self.input_size, self.input_size),
#                                 interpolation=cv2.INTER_AREA)
            
#         H, W, _ = img_lq2.shape
#         if H != self.input_size or W != self.input_size:
#             img_lq2 = cv2.resize(img_lq2, (self.input_size, self.input_size),
#                                 interpolation=cv2.INTER_AREA)
            
#         H, W, _ = img_gt1.shape
#         if H != self.input_size or W != self.input_size:
#             img_gt1 = cv2.resize(img_gt1, (self.input_size, self.input_size),
#                                 interpolation=cv2.INTER_AREA)
            
#         H, W, _ = img_gt2.shape
#         if H != self.input_size or W != self.input_size:
#             img_gt2 = cv2.resize(img_gt2, (self.input_size, self.input_size),
#                                 interpolation=cv2.INTER_AREA)
        
#         if img_gt1.ndim == 2:
#             img_gt1 = np.expand_dims(img_gt1, axis=2)
#             img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
#         if img_gt2.ndim == 2:
#             img_gt2 = np.expand_dims(img_gt2, axis=2)
#             img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
            
#         if img_gt1.shape[2] !=3:
#             img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
#         if img_gt2.shape[2] !=3:
#             img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
        
#         img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
#         img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
#         img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
#         img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
#         batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
#                  'input_query_img2': img_lq2, 'target_img2': img_gt2,
#                  'input_query_img2_path': lq2_path}
#         return batch, deg_type

# class DatasetPrompt_Customized_Test_DirectLoad_Triplet(Dataset):
#     def __init__(self, dataset_path_root, data_len=None):
#         self.dataset_path_root = dataset_path_root
#         self.dataset_paths = os.listdir(self.dataset_path_root)
#         sorted(self.dataset_paths)
#         self.data_len = data_len
        
#         random.seed(1000)
#         if self.data_len is not None:
#             random.shuffle(self.dataset_paths)
        
#     def __len__(self):
#         if self.data_len is not None:
#             return self.data_len
#         else: return len(self.dataset_paths)

#     def __getitem__(self, idx):
#         lq1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_input_img1.png')
#         gt1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_target_img1.png')
#         lq2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
#         if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
#             gt2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
#         else:
#             gt2_path = None
        
#         img_gt1 = util.read_img(None, gt1_path, None)
#         img_lq1 = util.read_img(None, lq1_path, None)
        
#         if gt2_path:
#             img_gt2 = util.read_img(None, gt2_path, None)
        
#         img_lq2 = util.read_img(None, lq2_path, None)
        
#         if img_gt1.ndim == 2:
#             img_gt1 = np.expand_dims(img_gt1, axis=2)
#             img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
#         if img_lq1.ndim == 2:
#             img_lq1 = np.expand_dims(img_lq1, axis=2)
#             img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
#         if gt2_path and img_gt2.ndim == 2:
#             img_gt2 = np.expand_dims(img_gt2, axis=2)
#             img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
#         if img_lq2.ndim == 2:
#             img_lq2 = np.expand_dims(img_lq2, axis=2)
#             img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)
            
#         if img_gt1.shape[2] !=3:
#             img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
#         if img_lq1.shape[2] !=3:
#             img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
#         if gt2_path and img_gt2.shape[2] !=3:
#             img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
#         if img_lq2.shape[2] !=3:
#             img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)

#         img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
#         if gt2_path:
#             img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
#         img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
#         img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
#         if gt2_path:
#             target_img2 = img_gt2
#         else:
#             target_img2 = 'None'
        
#         batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
#                  'input_query_img2': img_lq2, 'target_img2': target_img2,
#                  'input_query_img2_path': lq2_path}
        
#         deg_type = self.dataset_paths[idx]
#         return batch, deg_type
    
# if __name__ == "__main__":
#     dataset_train = DatasetPrompt_Train(dataset_path="data/imagenet1k/train", 
#                                         input_size=256,
#                                         CT_Covid_path='data/Restoration/CT_covid/train',
#                                         ITS_path='data/Restoration/Dehaze/ITS', 
#                                         Demoireing_path='data/Restoration/Demoireing_Processed/train',
#                                         Rain13K_path='data/Restoration/Deraining/Rain13K',
#                                         DustRemoval_path='data/Restoration/Dust_Removal/RB-Dust_processed/train', 
#                                         Desnowing_path='data/Restoration/Desnow_CSD/Train', 
#                                         FlareRemoval_path='data/Restoration/Flare7Kpp', 
#                                         FaceSR_path='data/Face/ffhq', 
#                                         HighlightRemoval_path='data/Restoration/highlight_removal/SHIQ_data_10825/train',
#                                         LensFlare_path='data/Restoration/lens-flare', 
#                                         LOL_path='data/Restoration/LOL_256/eval15', 
#                                         MRI_path = 'data/MRI/train',
#                                         MarineSnowRemovalSmall_path='data/Restoration/small_sized_marine_snow_removal',
#                                         MarineSnowRemovalVarious_path='data/Restoration/various_sized_marine_snow_removal',
#                                         ReflectionRemoval_path='data/Restoration/reflection_removal/zhang/synthetic',
#                                         Satellite_path='data/Restoration/satellite/train',
#                                         ShadowRemoval_ISTD_path='data/Restoration/Shadow_removal/ISTD_Dataset/train', 
#                                         ShadowRemoval_ISTD_adjusted_path='data/Restoration/Shadow_removal/ISTD_adjusted', 
#                                         ShadowRemoval_SRD_path='data/Restoration/Shadow_removal/SRD/Train',
#                                         ThinCloudRemoval_path='data/Restoration/thin_cloud_removal/train',
#                                         TextSR_train1_path='data/Restoration/TextSuperResolution-new/train/train1/png', 
#                                         TextSR_train2_path='data/Restoration/TextSuperResolution-new/train/train2/png',
#                                         Watermark_path='data/Restoration/watermark_removal',
#                                         RealLowLightSR_path='data/Restoration/Real_Lowlight_SR',
#                                         Backlit_path='data/Enhancement/backlit_image_enhancement/train/BAID_380',
#                                         DSLR_blackberry_path='data/uploaded_data/DSLR/dped/blackberry/training_data',
#                                         DSLR_iphone_path='data/uploaded_data/DSLR/dped/iphone/training_data',
#                                         DSLR_sony_path='data/uploaded_data/DSLR/dped/sony/training_data',
#                                         ExposureError_path='data/Enhancement/exposure_error/training', 
#                                         InstagramFilter_path='data/Enhancement/InstagramFilterRemoval/IFFI-dataset_resort/train', 
#                                         SDR_HDR_path='data/Enhancement/HDRTV1K/processed_sets', 
#                                         LLF_path='data/Enhancement/LLF_256', 
#                                         FiveK_path='data/Enhancement/MIT-fivek', 
#                                         Bokeh_path='data/Enhancement/Rendering_Realistic_Bokeh/EBBokeh_processed/train', 
#                                         Histo_Equ_path='data/Enhancement/UIEB_Dataset',
#                                         UDC_poled_path='data/Enhancement/UDC/train/Poled', 
#                                         UDC_toled_path='data/Enhancement/UDC/train/Toled',  
#                                         WhiteBalance_path='data/Enhancement/WhiteBalance', 
#                                         Color_Corre_path='data/Enhancement/UIEB_dive',
#                                         VignettingRemoval_path='data/Enhancement/vignetting_removal512/train', 
#                                         MultiTone_path='data/Enhancement/MIT-fivek',
#                                         ISP_path='data/Enhancement/ISP/train', 
#                                         Edge_Detect_path='data/Edge/BIPED/resize',
#                                         DepthEstimation_path='data/ImageTranslation/depth_estimation/train',
#                                         PencilDrawing_path='data/Enhancement/MIT-fivek', 
#                                         Photographic_path='data/Enhancement/MIT-fivek', 
#                                         RTV_path='data/Enhancement/MIT-fivek',
#                                         SaliencyObjectDetection_path='data/ImageTranslation/salient_object_detection/train', 
#                                         Style_Cloisonnism_path='data/Enhancement/MIT-fivek', 
#                                         Style_Divisionism_path='data/Enhancement/MIT-fivek', 
#                                         Style_Fauvism_path='data/Enhancement/MIT-fivek', 
#                                         Style_JOJO_path='data/Enhancement/MIT-fivek', 
#                                         Style_Vermeer_path='data/Enhancement/MIT-fivek', 
#                                         Style_Raphael_path='data/Enhancement/MIT-fivek', 
#                                         Style_Modernism_path='data/Enhancement/MIT-fivek', 
#                                         Style_Monet_path='data/Enhancement/MIT-fivek',
#                                         Style_NeoImpressionism_path='data/Enhancement/MIT-fivek', 
#                                         Style_PopArt_path='data/Enhancement/MIT-fivek', 
#                                         Style_Ukiyoe_path='data/Enhancement/MIT-fivek', 
#                                         Style_VanGogh_path='data/Enhancement/MIT-fivek',
#                                         data_len=None
#                                         )

#     data_loader_train = torch.utils.data.DataLoader(
#         dataset_train,
#         batch_size=1,
#         num_workers=6,
#         drop_last=True,
#     )
    
#     for idx, inp_data in enumerate(data_loader_train):
#         batch, degtype = inp_data
#         print(degtype)
    