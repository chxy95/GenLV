import os
import numpy as np
import torch
from torch.utils.data import Dataset
from evaluate.add_degradation_various import *
from dataset.image_operators import *
from dataset.add_flare import add_flare, add_lens_flare
from dataset.add_reflection import add_reflection
from dataset.x_distortion import *
from glob import glob

import sys
import dataset.util as util

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
        img_gt1 = Laplacian_edge_detector_uint8(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = Laplacian_edge_detector_uint8(img_gt2.copy())
    elif deg_type == 'Canny':
        img_lq1 = img_gt1.copy()
        img_gt1 = Canny_edge_detector_uint8(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = Canny_edge_detector_uint8(img_gt2.copy())
    # check zero images.
    if np.mean(img_gt1).astype(np.float16) == 0 or np.mean(img_gt2).astype(np.float16) == 0:
        if np.mean(img_gt1).astype(np.float16) == 0 and np.mean(img_gt2).astype(np.float16) == 0:
            print(deg_type, 'prompt&query zero images.')
        if np.mean(img_gt1).astype(np.float16) == 0:
            print(deg_type, 'prompt gt zero image.')
            img_gt1 = img_gt2.copy()
            img_lq1 = img_lq2.copy()
        if np.mean(img_gt2).astype(np.float16) == 0:
            print(deg_type, 'query gt zero image.')
            img_gt2 = img_gt1.copy()
            img_lq2 = img_lq1.copy()
    return img_lq1, img_lq2, img_gt1, img_gt2 

def add_degradation_two_images(img_gt1, img_gt2, deg_type):
    # np.float32
    if deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq1 = add_Gaussian_noise(img_gt1.copy(), level=level)
        level = random.uniform(10, 50)
        img_lq2 = add_Gaussian_noise(img_gt2.copy(), level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq1 = iso_GaussianBlur(img_gt1.copy(), window=15, sigma=sigma)
        sigma = random.uniform(2, 4)
        img_lq2 = iso_GaussianBlur(img_gt2.copy(), window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq1 = add_JPEG_noise(img_gt1.copy(), level=level)
        level = random.randint(10, 40)
        img_lq2 = add_JPEG_noise(img_gt2.copy(), level=level)
    elif deg_type == 'Resize':
        img_lq1 = add_resize(img_gt1.copy())
        img_lq2 = add_resize(img_gt2.copy())
    elif deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq1 = add_rain(img_gt1.copy(), value=value)
        value = random.uniform(40, 200)
        img_lq2 = add_rain(img_gt2.copy(), value=value)
    elif deg_type == 'SPNoise':
        img_lq1 = add_sp_noise(img_gt1.copy())
        img_lq2 = add_sp_noise(img_gt2.copy())
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq1 = low_light(img_gt1.copy(), lum_scale=lum_scale)
        img_lq2 = low_light(img_gt2.copy(), lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq1 = add_Poisson_noise(img_gt1.copy(), level=2)
        img_lq2 = add_Poisson_noise(img_gt2.copy(), level=2)
    elif deg_type == 'Ringing':
        img_lq1 = add_ringing(img_gt1.copy())
        img_lq2 = add_ringing(img_gt2.copy())
    elif deg_type == 'r_l':
        img_lq1 = r_l(img_gt1.copy())
        img_lq2 = r_l(img_gt2.copy())
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_lq1 = inpainting(img_gt1.copy(), l_num=l_num, l_thick=l_thick)
        img_lq2 = inpainting(img_gt2.copy(), l_num=l_num, l_thick=l_thick)
    elif deg_type == 'gray':
        img_lq1 = cv2.cvtColor(img_gt1.copy(), cv2.COLOR_BGR2GRAY)
        img_lq1 = np.expand_dims(img_lq1, axis=2)
        img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
        img_lq2 = cv2.cvtColor(img_gt2.copy(), cv2.COLOR_BGR2GRAY)
        img_lq2 = np.expand_dims(img_lq2, axis=2)
        img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)
    elif deg_type == 'Defocus_Blur':
        img_lq1 = defocus_blur(img_gt1.copy())
        img_lq2 = defocus_blur(img_gt2.copy())
    elif deg_type == 'mosaic':
        img_lq1 = mosaic_CFA_Bayer(img_gt1.copy())
        img_lq2 = mosaic_CFA_Bayer(img_gt2.copy())
    # elif deg_type == 'L0_smooth':
    #     img_lq1 = img_gt1.copy()
    #     img_gt1 = L0_smooth(img_gt1.copy())
    #     img_lq2 = img_gt2.copy()
    #     img_gt2 = L0_smooth(img_gt2.copy())
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
    def __init__(self, dataset_path, input_size, CT_Covid_path=None, ITS_path=None, Demoireing_path=None,
                 Desnowing_path=None, FlareRemoval_path=None, Rain13K_path=None, DustRemoval_path=None,
                 FaceSR_path=None, HighlightRemoval_path=None, LensFlare_path=None, LOL_path=None, MRI_path=None, MarineSnowRemovalSmall_path=None, MarineSnowRemovalVarious_path=None,
                 ReflectionRemoval_path=None, RealLowLightSR_path=None, Satellite_path=None, ShadowRemoval_ISTD_path=None, ShadowRemoval_ISTD_adjusted_path=None, 
                 ShadowRemoval_SRD_path=None, TextSR_train1_path=None, TextSR_train2_path=None, ThinCloudRemoval_path=None, Watermark_path=None,
                 Backlit_path=None, ExposureError_path=None, InstagramFilter_path=None, ISP_path=None, SDR_HDR_path=None, 
                 DSLR_blackberry_path=None, DSLR_iphone_path=None, DSLR_sony_path=None, LLF_path=None, FiveK_path=None, RedEye_path=None, UDC_poled_path=None, 
                 UDC_toled_path=None, Bokeh_path=None, Histo_Equ_path=None, VignettingRemoval_path=None,
                 WhiteBalance_path=None, Color_Corre_path=None, MultiTone_path=None, 
                 Edge_Detect_path=None, PencilDrawing_path=None, Photographic_path=None, DepthEstimation_path=None, SaliencyObjectDetection_path=None,
                 RTV_path=None, Style_Cloisonnism_path=None, Style_Divisionism_path=None, 
                 Style_Fauvism_path=None, Style_JOJO_path=None, Style_Vermeer_path=None, 
                 Style_Raphael_path=None, Style_Modernism_path=None, Style_Monet_path=None,
                 Style_NeoImpressionism_path=None, Style_PopArt_path=None, Style_Ukiyoe_path=None, 
                 Style_VanGogh_path=None, data_len=None, tasks_version=0
                 ):
        
        self.data_len = data_len
        self.tasks_version = tasks_version
        
        np.random.seed(5)
        self.gt_size = input_size
        
        # base dataset
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.paths_base_len = len(self.paths_gt)
        
        ##########################################################################
        # Restoration
        
        # CT covid dataset
        # data/Restoration/CT_covid
        if CT_Covid_path is not None:
            self.dataset_path_gt_CT_Covid = CT_Covid_path
            self.paths_gt_CT_Covid, self.sizes_gt_CT_Covid = util.get_image_paths('img', self.dataset_path_gt_CT_Covid)
            self.paths_CT_Covid_len = len(self.paths_gt_CT_Covid)
            sorted(self.paths_gt_CT_Covid)
            
        # Dehaze dataset (ITS)
        # data/Restoration/Dehaze/ITS
        if ITS_path is not None:
            self.dataset_path_gt_ITS = os.path.join(ITS_path, 'clear')
            self.dataset_path_lq_ITS = os.path.join(ITS_path, 'hazy')
            self.paths_gt_ITS, self.sizes_gt_ITS = util.get_image_paths('img', self.dataset_path_gt_ITS)
            self.paths_lq_ITS, self.sizes_lq_ITS = util.get_image_paths('img', self.dataset_path_lq_ITS)
            self.paths_ITS_len = len(self.paths_lq_ITS)
        
        # Demoireing dataset
        # data/Restoration/Demoireing_Processed/train
        if Demoireing_path is not None:
            self.dataset_path_gt_Demoireing = os.path.join(Demoireing_path, 'gt')
            self.dataset_path_lq_Demoireing = os.path.join(Demoireing_path, 'lq')
            self.paths_gt_Demoireing, self.sizes_gt_Demoireing = util.get_image_paths('img', self.dataset_path_gt_Demoireing)
            self.paths_lq_Demoireing, self.sizes_lq_Demoireing = util.get_image_paths('img', self.dataset_path_lq_Demoireing)
            self.paths_Demoireing_len = len(self.paths_lq_Demoireing)
            sorted(self.paths_gt_Demoireing)
            sorted(self.paths_lq_Demoireing)
            
        # Derain dataset (Rain13K)
        # data/Restoration/Deraining/Rain13K
        if Rain13K_path is not None:
            self.dataset_path_gt_Rain13K = os.path.join(Rain13K_path, 'target')
            self.dataset_path_lq_Rain13K = os.path.join(Rain13K_path, 'input')
            self.paths_gt_Rain13K, self.sizes_gt_Rain13K = util.get_image_paths('img', self.dataset_path_gt_Rain13K)
            self.paths_lq_Rain13K, self.sizes_lq_Rain13K = util.get_image_paths('img', self.dataset_path_lq_Rain13K)
            self.paths_Rain13K_len = len(self.paths_lq_Rain13K)
            sorted(self.paths_gt_Rain13K)
            sorted(self.paths_lq_Rain13K)
        
        # Dust Removal dataset
        # data/Restoration/Dust_Removal/RB-Dust_processed/train
        if DustRemoval_path is not None:
            self.dataset_path_gt_DustRemoval = os.path.join(DustRemoval_path, 'gt')
            self.dataset_path_lq_DustRemoval = os.path.join(DustRemoval_path, 'lq')
            self.paths_gt_DustRemoval, self.sizes_gt_DustRemoval = util.get_image_paths('img', self.dataset_path_gt_DustRemoval)
            self.paths_lq_DustRemoval, self.sizes_lq_DustRemoval = util.get_image_paths('img', self.dataset_path_lq_DustRemoval)
            self.paths_DustRemoval_len = len(self.paths_lq_DustRemoval)
            sorted(self.paths_gt_DustRemoval)
            sorted(self.paths_lq_DustRemoval)
        
        # Desnowing dataset
        # data/Restoration/Desnow_CSD/Train
        if Desnowing_path is not None:
            self.dataset_path_gt_Desnowing = os.path.join(Desnowing_path, 'gt')
            self.dataset_path_lq_Desnowing = os.path.join(Desnowing_path, 'lq')
            self.paths_gt_Desnowing, self.sizes_gt_Desnowing = util.get_image_paths('img', self.dataset_path_gt_Desnowing)
            self.paths_lq_Desnowing, self.sizes_lq_Desnowing = util.get_image_paths('img', self.dataset_path_lq_Desnowing)
            self.paths_Desnowing_len = len(self.paths_lq_Desnowing)
            sorted(self.paths_gt_Desnowing)
            sorted(self.paths_lq_Desnowing)
        
        # Flare7K dataset, generate images on the fly according to /cpfs01/user/puyuandong/glv/Flare7k
        # data/Restoration/Flare7Kpp
        if FlareRemoval_path is not None:
            self.dataset_path_gt_FlareRemoval = os.path.join(FlareRemoval_path, 'Flickr24K')
            self.dataset_path_scattering_synthesis = os.path.join(FlareRemoval_path, 'Flare7K', 'Scattering_Flare', 'Compound_Flare')
            self.dataset_path_scattering_real = os.path.join(FlareRemoval_path, 'Flare-R', 'Compound_Flare')
            self.dataset_path_reflective = os.path.join(FlareRemoval_path, 'Flare7K', 'Reflective_Flare')
            self.dataset_path_light_synthesis = os.path.join(FlareRemoval_path, 'Flare7K', 'Scattering_Flare', 'Light_Source')
            self.dataset_path_light_real = os.path.join(FlareRemoval_path, 'Flare-R', 'Light_Source')
            
            self.paths_gt_FlareRemoval, self.sizes_gt_FlareRemoval = util.get_image_paths('img', self.dataset_path_gt_FlareRemoval)
            self.paths_scattering_synthesis, self.sizes_scattering_synthesis = util.get_image_paths('img', self.dataset_path_scattering_synthesis)
            self.paths_scattering_real, self.sizes_scattering_real = util.get_image_paths('img', self.dataset_path_scattering_real)
            self.paths_reflective, self.sizes_reflective = util.get_image_paths('img', self.dataset_path_reflective)
            self.paths_light_synthesis, self.sizes_light_synthesis = util.get_image_paths('img', self.dataset_path_light_synthesis)
            self.paths_light_real, self.sizes_light_real = util.get_image_paths('img', self.dataset_path_light_real)
            
            self.paths_FlareRemoval_len = len(self.paths_gt_FlareRemoval)
            sorted(self.paths_gt_FlareRemoval)
            sorted(self.paths_scattering_synthesis)
            sorted(self.paths_scattering_real)
            sorted(self.paths_reflective)
            sorted(self.paths_light_synthesis)
            sorted(self.paths_light_real)
        
        # Face dataset
        # data/Face/ffhq
        if FaceSR_path is not None:
            self.dataset_path_gt_FaceSR = os.path.join(FaceSR_path, 'ffhq512_images')
            self.paths_gt_FaceSR, self.sizes_gt_FaceSR = util.get_image_paths('img', self.dataset_path_gt_FaceSR)
            self.paths_FaceSR_len = len(self.paths_gt_FaceSR)
            sorted(self.paths_gt_FaceSR)
        
        # Highlight Removal dataset
        # data/Restoration/highlight_removal/SHIQ_data_10825/train
        if HighlightRemoval_path is not None:
            self.dataset_path_gt_HighlightRemoval = os.path.join(HighlightRemoval_path, 'gt')
            self.dataset_path_lq_HighlightRemoval = os.path.join(HighlightRemoval_path, 'lq')
            self.paths_gt_HighlightRemoval, self.sizes_gt_HighlightRemoval = util.get_image_paths('img', self.dataset_path_gt_HighlightRemoval)
            self.paths_lq_HighlightRemoval, self.sizes_lq_HighlightRemoval = util.get_image_paths('img', self.dataset_path_lq_HighlightRemoval)
            self.paths_HighlightRemoval_len = len(self.paths_lq_HighlightRemoval)
            sorted(self.paths_gt_HighlightRemoval)
            sorted(self.paths_lq_HighlightRemoval)
        
        # LensFlare dataset
        # data/Restoration/lens-flare
        if LensFlare_path is not None:
            self.dataset_path_gt_LensFlare = os.path.join(LensFlare_path, 'Flickr24K')
            self.dataset_path_flare_captured = os.path.join(LensFlare_path, 'captured')
            self.dataset_path_flare_simulated = os.path.join(LensFlare_path, 'simulated')
            
            self.paths_gt_LensFlare, self.sizes_gt_LensFlare = util.get_image_paths('img', self.dataset_path_gt_LensFlare)
            self.paths_flare_captured, self.sizes_flare_captured = util.get_image_paths('img', self.dataset_path_flare_captured)
            self.paths_flare_simulated, self.sizes_flare_simulated = util.get_image_paths('img', self.dataset_path_flare_simulated)
            
            self.paths_LensFlare_len = len(self.paths_gt_LensFlare)
            
            sorted(self.paths_gt_LensFlare)
            sorted(self.paths_flare_captured)
            sorted(self.paths_flare_simulated)
            
        # low-light enhancement dataset (LOL)
        # data/Restoration/LOL_256/eval15
        if LOL_path is not None:
            self.dataset_path_gt_LOL = os.path.join(LOL_path, 'high')
            self.dataset_path_lq_LOL = os.path.join(LOL_path, 'low')
            self.paths_gt_LOL, self.sizes_gt_LOL = util.get_image_paths('img', self.dataset_path_gt_LOL)
            self.paths_lq_LOL, self.sizes_lq_LOL = util.get_image_paths('img', self.dataset_path_lq_LOL)
            self.paths_LOL_len = len(self.paths_lq_LOL)
            sorted(self.paths_gt_LOL)
            sorted(self.paths_lq_LOL)
            
        # MRI dataset
        # data/MRI
        if MRI_path is not None:
            self.dataset_path_gt_MRI = MRI_path
            self.paths_gt_MRI, self.sizes_gt_MRI = util.get_image_paths('img', self.dataset_path_gt_MRI)
            self.paths_MRI_len = len(self.paths_gt_MRI)
            sorted(self.paths_gt_MRI)
        
        if MarineSnowRemovalSmall_path is not None:
            self.dataset_path_gt_MarineSnowRemovalSmall = os.path.join(MarineSnowRemovalSmall_path, 'gt')
            self.dataset_path_lq_MarineSnowRemovalSmall = os.path.join(MarineSnowRemovalSmall_path, 'lq')
            self.paths_gt_MarineSnowRemovalSmall, self.sizes_gt_MarineSnowRemovalSmall = util.get_image_paths('img', self.dataset_path_gt_MarineSnowRemovalSmall)
            self.paths_lq_MarineSnowRemovalSmall, self.sizes_lq_MarineSnowRemovalSmall = util.get_image_paths('img', self.dataset_path_lq_MarineSnowRemovalSmall)
            self.paths_MarineSnowRemovalSmall_len = len(self.paths_lq_MarineSnowRemovalSmall)
            sorted(self.paths_gt_MarineSnowRemovalSmall)
            sorted(self.paths_lq_MarineSnowRemovalSmall)
        
        if MarineSnowRemovalVarious_path is not None:
            self.dataset_path_gt_MarineSnowRemovalVarious = os.path.join(MarineSnowRemovalVarious_path, 'gt')
            self.dataset_path_lq_MarineSnowRemovalVarious = os.path.join(MarineSnowRemovalVarious_path, 'lq')
            self.paths_gt_MarineSnowRemovalVarious, self.sizes_gt_MarineSnowRemovalVarious = util.get_image_paths('img', self.dataset_path_gt_MarineSnowRemovalVarious)
            self.paths_lq_MarineSnowRemovalVarious, self.sizes_lq_MarineSnowRemovalVarious = util.get_image_paths('img', self.dataset_path_lq_MarineSnowRemovalVarious)
            self.paths_MarineSnowRemovalVarious_len = len(self.paths_lq_MarineSnowRemovalVarious)
            sorted(self.paths_gt_MarineSnowRemovalVarious)
            sorted(self.paths_lq_MarineSnowRemovalVarious)
            
        # Reflection Removal dataset
        # data/Restoration/reflection_removal/zhang/synthetic
        if ReflectionRemoval_path is not None:
            self.dataset_path_transmission_ReflectionRemoval = os.path.join(ReflectionRemoval_path, 'transmission_layer')
            self.dataset_path_reflection_ReflectionRemoval = os.path.join(ReflectionRemoval_path, 'reflection_layer')
            self.paths_transmission_ReflectionRemoval, self.sizes_transmission_ReflectionRemoval = util.get_image_paths('img', self.dataset_path_transmission_ReflectionRemoval)
            self.paths_reflection_ReflectionRemoval, self.sizes_reflection_ReflectionRemoval = util.get_image_paths('img', self.dataset_path_reflection_ReflectionRemoval)
            self.paths_reflection_ReflectionRemoval_len = len(self.paths_reflection_ReflectionRemoval)
            self.paths_transmission_ReflectionRemoval_len = len(self.paths_transmission_ReflectionRemoval)
            sorted(self.paths_transmission_ReflectionRemoval)
            sorted(self.paths_reflection_ReflectionRemoval)
        
        # Satellite dataset
        # data/Restoration/satellite/train
        if Satellite_path is not None:
            self.dataset_path_gt_Satellite = Satellite_path
            self.paths_gt_Satellite, self.sizes_gt_Satellite = util.get_image_paths('img', self.dataset_path_gt_Satellite)
            self.paths_Satellite_len = len(self.paths_gt_Satellite)
            sorted(self.paths_gt_Satellite)
            
        # Shadow Removal dataset
        # data/Restoration/Shadow_removal/ISTD_Dataset/train
        if ShadowRemoval_ISTD_path is not None:
            self.dataset_path_gt_ShadowRemoval_ISTD = os.path.join(ShadowRemoval_ISTD_path, 'train_C')
            self.dataset_path_lq_ShadowRemoval_ISTD = os.path.join(ShadowRemoval_ISTD_path, 'train_A')
            self.paths_gt_ShadowRemoval_ISTD, self.sizes_gt_ShadowRemoval_ISTD = util.get_image_paths('img', self.dataset_path_gt_ShadowRemoval_ISTD)
            self.paths_lq_ShadowRemoval_ISTD, self.sizes_lq_ShadowRemoval_ISTD = util.get_image_paths('img', self.dataset_path_lq_ShadowRemoval_ISTD)
            self.paths_ShadowRemoval_ISTD_len = len(self.paths_lq_ShadowRemoval_ISTD)
            sorted(self.paths_gt_ShadowRemoval_ISTD)
            sorted(self.paths_lq_ShadowRemoval_ISTD)
        
        # data/Restoration/Shadow_removal/ISTD_adjusted
        if ShadowRemoval_ISTD_adjusted_path is not None:
            self.dataset_path_gt_ShadowRemoval_ISTD_adjusted = os.path.join(ShadowRemoval_ISTD_adjusted_path, 'train_C_fixed_ours')
            self.dataset_path_lq_ShadowRemoval_ISTD_adjusted = os.path.join(ShadowRemoval_ISTD_adjusted_path, 'train_A')
            self.paths_gt_ShadowRemoval_ISTD_adjusted, self.sizes_gt_ShadowRemoval_ISTD_adjusted = util.get_image_paths('img', self.dataset_path_gt_ShadowRemoval_ISTD_adjusted)
            self.paths_lq_ShadowRemoval_ISTD_adjusted, self.sizes_lq_ShadowRemoval_ISTD_adjusted = util.get_image_paths('img', self.dataset_path_lq_ShadowRemoval_ISTD_adjusted)
            self.paths_ShadowRemoval_ISTD_adjusted_len = len(self.paths_lq_ShadowRemoval_ISTD_adjusted)
            sorted(self.paths_gt_ShadowRemoval_ISTD_adjusted)
            sorted(self.paths_lq_ShadowRemoval_ISTD_adjusted)
        
        # data/Restoration/Shadow_removal/SRD/Train
        if ShadowRemoval_SRD_path is not None:
            self.dataset_path_gt_ShadowRemoval_SRD = os.path.join(ShadowRemoval_SRD_path, 'shadow_free')
            self.dataset_path_lq_ShadowRemoval_SRD = os.path.join(ShadowRemoval_SRD_path, 'shadow')
            self.paths_gt_ShadowRemoval_SRD, self.sizes_gt_ShadowRemoval_SRD = util.get_image_paths('img', self.dataset_path_gt_ShadowRemoval_SRD)
            self.paths_lq_ShadowRemoval_SRD, self.sizes_lq_ShadowRemoval_SRD = util.get_image_paths('img', self.dataset_path_lq_ShadowRemoval_SRD)
            self.paths_ShadowRemoval_SRD_len = len(self.paths_lq_ShadowRemoval_SRD)
            sorted(self.paths_gt_ShadowRemoval_SRD)
            sorted(self.paths_lq_ShadowRemoval_SRD)
        
        # data/Restoration/TextSuperResolution-new/train/train1/png
        if TextSR_train1_path is not None:
            self.dataset_path_gt_TextSR_train1 = os.path.join(TextSR_train1_path, 'gt')
            self.dataset_path_lq_TextSR_train1 = os.path.join(TextSR_train1_path, 'lq')
            self.paths_gt_TextSR_train1, self.sizes_gt_train1 = util.get_image_paths('img', self.dataset_path_gt_TextSR_train1)
            self.paths_lq_TextSR_train1, self.sizes_lq_train1 = util.get_image_paths('img', self.dataset_path_lq_TextSR_train1)
            self.paths_TextSR_train1_len = len(self.paths_lq_TextSR_train1)
            sorted(self.paths_gt_TextSR_train1)
            sorted(self.paths_lq_TextSR_train1)
        
        # data/Restoration/TextSuperResolution-new/train/train2/png
        if TextSR_train2_path is not None:
            self.dataset_path_gt_TextSR_train2 = os.path.join(TextSR_train2_path, 'gt')
            self.dataset_path_lq_TextSR_train2 = os.path.join(TextSR_train2_path, 'lq')
            self.paths_gt_TextSR_train2, self.sizes_gt_TextSR_train2 = util.get_image_paths('img', self.dataset_path_gt_TextSR_train2)
            self.paths_lq_TextSR_train2, self.sizes_lq_TextSR_train2 = util.get_image_paths('img', self.dataset_path_lq_TextSR_train2)
            self.paths_TextSR_train2_len = len(self.paths_lq_TextSR_train2)
            sorted(self.paths_gt_TextSR_train2)
            sorted(self.paths_lq_TextSR_train2)
        
        # data/Restoration/thin_cloud_removal
        if ThinCloudRemoval_path is not None:
            self.dataset_path_gt_ThinCloudRemoval = os.path.join(ThinCloudRemoval_path, 'gt')
            self.dataset_path_lq_ThinCloudRemoval = os.path.join(ThinCloudRemoval_path, 'lq')
            self.paths_gt_ThinCloudRemoval, self.sizes_gt_ThinCloudRemoval = util.get_image_paths('img', self.dataset_path_gt_ThinCloudRemoval)
            self.paths_lq_ThinCloudRemoval, self.sizes_lq_ThinCloudRemoval = util.get_image_paths('img', self.dataset_path_lq_ThinCloudRemoval)
            self.paths_ThinCloudRemoval_len = len(self.paths_lq_ThinCloudRemoval)
            sorted(self.paths_gt_ThinCloudRemoval)
            sorted(self.paths_lq_ThinCloudRemoval)
            
        # data/Restoration/watermark_removal
        if Watermark_path is not None:
            self.dataset_path_gt_Watermark = os.path.join(Watermark_path, 'gt')
            self.dataset_path_lq_Watermark = os.path.join(Watermark_path, 'lq')
            self.paths_gt_Watermark, self.sizes_gt_Watermark = util.get_image_paths('img', self.dataset_path_gt_Watermark)
            self.paths_lq_Watermark, self.sizes_lq_Watermark = util.get_image_paths('img', self.dataset_path_lq_Watermark)
            self.paths_Watermark_len = len(self.paths_lq_Watermark)
            sorted(self.paths_gt_Watermark)
            sorted(self.paths_lq_Watermark)
        
        if RealLowLightSR_path is not None:
            self.dataset_path_gt_RealLowLightSR = os.path.join(RealLowLightSR_path, 'NLHR/X4')
            self.dataset_path_lq_RealLowLightSR = os.path.join(RealLowLightSR_path, 'LLLR')
            self.paths_gt_RealLowLightSR, self.sizes_gt_RealLowLightSR = util.get_image_paths('img', self.dataset_path_gt_RealLowLightSR)
            self.paths_lq_RealLowLightSR, self.sizes_lq_RealLowLightSR = util.get_image_paths('img', self.dataset_path_lq_RealLowLightSR)
            self.paths_RealLowLightSR_len = len(self.paths_lq_RealLowLightSR)
            sorted(self.paths_gt_RealLowLightSR)
            sorted(self.paths_lq_RealLowLightSR)
            
        ##########################################################################
        # Enhancement
        # Backlit Image Enhancement dataset
        # data/Enhancement/backlit_image_enhancement/train/BAID_380
        if Backlit_path is not None:
            self.dataset_path_gt_Backlit = os.path.join(Backlit_path, 'resize_gt')
            self.dataset_path_lq_Backlit = os.path.join(Backlit_path, 'resize_input')
            self.paths_gt_Backlit, self.sizes_gt_Backlit = util.get_image_paths('img', self.dataset_path_gt_Backlit)
            self.paths_lq_Backlit, self.sizes_lq_Backlit = util.get_image_paths('img', self.dataset_path_lq_Backlit)
            self.paths_Backlit_len = len(self.paths_lq_Backlit)
            sorted(self.paths_gt_Backlit)
            sorted(self.paths_lq_Backlit)
            
        if DSLR_blackberry_path is not None:
            self.dataset_path_gt_DSLR_blackberry = os.path.join(DSLR_blackberry_path, 'canon')
            self.dataset_path_lq_DSLR_blackberry = os.path.join(DSLR_blackberry_path, 'blackberry')
            self.paths_gt_DSLR_blackberry, self.sizes_gt_DSLR_blackberry = util.get_image_paths('img', self.dataset_path_gt_DSLR_blackberry)
            self.paths_lq_DSLR_blackberry, self.sizes_lq_DSLR_blackberry = util.get_image_paths('img', self.dataset_path_lq_DSLR_blackberry)
            self.paths_DSLR_blackberry_len = len(self.paths_lq_DSLR_blackberry)
            sorted(self.paths_gt_DSLR_blackberry)
            sorted(self.paths_lq_DSLR_blackberry)
            
        if DSLR_iphone_path is not None:
            self.dataset_path_gt_DSLR_iphone = os.path.join(DSLR_iphone_path, 'canon')
            self.dataset_path_lq_DSLR_iphone = os.path.join(DSLR_iphone_path, 'iphone')
            self.paths_gt_DSLR_iphone, self.sizes_gt_DSLR_iphone = util.get_image_paths('img', self.dataset_path_gt_DSLR_iphone)
            self.paths_lq_DSLR_iphone, self.sizes_lq_DSLR_iphone = util.get_image_paths('img', self.dataset_path_lq_DSLR_iphone)
            self.paths_DSLR_iphone_len = len(self.paths_lq_DSLR_iphone)
            sorted(self.paths_gt_DSLR_iphone)
            sorted(self.paths_lq_DSLR_iphone)
            
        if DSLR_sony_path is not None:
            self.dataset_path_gt_DSLR_sony = os.path.join(DSLR_sony_path, 'canon')
            self.dataset_path_lq_DSLR_sony = os.path.join(DSLR_sony_path, 'sony')
            self.paths_gt_DSLR_sony, self.sizes_gt_DSLR_sony = util.get_image_paths('img', self.dataset_path_gt_DSLR_sony)
            self.paths_lq_DSLR_sony, self.sizes_lq_DSLR_sony = util.get_image_paths('img', self.dataset_path_lq_DSLR_sony)
            self.paths_DSLR_sony_len = len(self.paths_lq_DSLR_sony)
            sorted(self.paths_gt_DSLR_sony)
            sorted(self.paths_lq_DSLR_sony)
            
        # Exposure Error dataset
        # data/Enhancement/exposure_error/training
        if ExposureError_path is not None:
            #To Do: Multiple lq images to one gt image
            self.dataset_path_gt_ExposureError = os.path.join(ExposureError_path, 'GT_IMAGES')
            self.dataset_path_lq_ExposureError = os.path.join(ExposureError_path, 'INPUT_IMAGES')
            self.paths_gt_ExposureError, self.sizes_gt_ExposureError = util.get_image_paths('img', self.dataset_path_gt_ExposureError)
            self.paths_lq_ExposureError, self.sizes_lq_ExposureError = util.get_image_paths('img', self.dataset_path_lq_ExposureError)
            self.paths_lq_ExposureError_len = len(self.paths_lq_ExposureError)
            self.paths_gt_ExposureError_len = len(self.paths_gt_ExposureError)
            sorted(self.paths_gt_ExposureError)
            sorted(self.paths_lq_ExposureError)
        
        # Instagram Filter Removal dataset
        # data/Enhancement/InstagramFilterRemoval/IFFI-dataset_resort/train
        if InstagramFilter_path is not None:
            # TO DO: Different folder structure
            self.dataset_path_ori_InstagramFilter = os.path.join(InstagramFilter_path, 'gt')
            self.dataset_path_enh_InstagramFilter = os.path.join(InstagramFilter_path, 'lq')
            self.paths_ori_InstagramFilter, self.sizes_ori_InstagramFilter = util.get_image_paths('img', self.dataset_path_ori_InstagramFilter)
            self.paths_enh_InstagramFilter, self.sizes_enh_InstagramFilter = util.get_image_paths('img', self.dataset_path_enh_InstagramFilter)
            self.ori_paths_InstagramFilter_len = len(self.paths_ori_InstagramFilter)
            self.enh_paths_InstagramFilter_len = len(self.paths_enh_InstagramFilter)
            sorted(self.paths_ori_InstagramFilter)
            sorted(self.paths_enh_InstagramFilter)
        
        # SDH-HDR dataset 
        # data/Enhancement/HDRTV1K/processed_sets
        if SDR_HDR_path is not None:
            self.dataset_path_SDR = os.path.join(SDR_HDR_path, 'resize_train_sdr')
            self.dataset_path_HDR = os.path.join(SDR_HDR_path, 'resize_train_hdr_quant')
            self.paths_SDR, self.sizes_SDR = util.get_image_paths('img', self.dataset_path_SDR)
            self.paths_HDR, self.sizes_HDR = util.get_image_paths('img', self.dataset_path_HDR)
            self.paths_SDR_len = len(self.paths_SDR)
            sorted(self.paths_SDR)
            sorted(self.paths_HDR)
        
        
        # Local Laplacian Filter dataset (MIT-Adobe FiveK - LLF)
        # data/Enhancement/LLF_256
        if LLF_path is not None:
            self.dataset_path_gt_LLF = os.path.join(LLF_path, 'gt')
            self.dataset_path_lq_LLF = os.path.join(LLF_path, 'input')
            self.paths_gt_LLF, self.sizes_gt_LLF = util.get_image_paths('img', self.dataset_path_gt_LLF)
            self.paths_lq_LLF, self.sizes_lq_LLF = util.get_image_paths('img', self.dataset_path_lq_LLF)
            self.paths_LLF_len = len(self.paths_lq_LLF)
            sorted(self.paths_gt_LLF)
            sorted(self.paths_lq_LLF)
        
        # Image Retouching dataset (MIT-Adobe FiveK)
        # data/Enhancement/MIT-fivek
        if FiveK_path is not None:
            self.dataset_path_gt_FiveK = os.path.join(FiveK_path, 'expert_C_train')
            self.dataset_path_lq_FiveK = os.path.join(FiveK_path, 'raw_input_train_png')
            self.paths_gt_FiveK, self.sizes_gt_FiveK = util.get_image_paths('img', self.dataset_path_gt_FiveK)
            self.paths_lq_FiveK, self.sizes_lq_FiveK = util.get_image_paths('img', self.dataset_path_lq_FiveK)
            self.paths_FiveK_len = len(self.paths_lq_FiveK)
            sorted(self.paths_gt_FiveK)
            sorted(self.paths_lq_FiveK)
            
        # Rendering Realistic Bokeh dataset
        # data/Enhancement/Rendering_Realistic_Bokeh/EBBokeh_processed/train
        if Bokeh_path is not None:
            self.dataset_path_gt_Bokeh = os.path.join(Bokeh_path, 'bokeh')
            self.dataset_path_lq_Bokeh = os.path.join(Bokeh_path, 'original')
            self.paths_gt_Bokeh, self.sizes_gt_Bokeh = util.get_image_paths('img', self.dataset_path_gt_Bokeh)
            self.paths_lq_Bokeh, self.sizes_lq_Bokeh = util.get_image_paths('img', self.dataset_path_lq_Bokeh)
            self.paths_Bokeh_len = len(self.paths_lq_Bokeh)
            sorted(self.paths_gt_Bokeh)
            sorted(self.paths_lq_Bokeh)

        # UIEB dataset (UIEBD - Histogram Equalization)
        # data/Enhancement/UIEB_Dataset
        if Histo_Equ_path is not None:
            self.dataset_path_gt_Histo_Equ = os.path.join(Histo_Equ_path, 'resize')
            self.dataset_path_lq_Histo_Equ = os.path.join(Histo_Equ_path, 'resize_histogram_equalization')
            self.paths_gt_Histo_Equ, self.sizes_gt_Histo_Equ = util.get_image_paths('img', self.dataset_path_gt_Histo_Equ)
            self.paths_lq_Histo_Equ, self.sizes_lq_Histo_Equ = util.get_image_paths('img', self.dataset_path_lq_Histo_Equ)
            self.paths_Histo_Equ_len = len(self.paths_lq_Histo_Equ)
            sorted(self.paths_gt_Histo_Equ)
            sorted(self.paths_lq_Histo_Equ)
        
        # UDC dataset
        # data/Enhancement/UDC/train/Poled
        if UDC_poled_path is not None:
            self.dataset_path_gt_UDC_poled = os.path.join(UDC_poled_path, 'HQ_256')
            self.dataset_path_lq_UDC_poled = os.path.join(UDC_poled_path, 'LQ_256')
            self.paths_gt_UDC_poled, self.sizes_gt_UDC_poled = util.get_image_paths('img', self.dataset_path_gt_UDC_poled)
            self.paths_lq_UDC_poled, self.sizes_lq_UDC_poled = util.get_image_paths('img', self.dataset_path_lq_UDC_poled)
            self.paths_UDC_poled_len = len(self.paths_lq_UDC_poled)
            sorted(self.paths_gt_UDC_poled)
            sorted(self.paths_lq_UDC_poled)
        
        # data/Enhancement/UDC/train/Toled
        if UDC_toled_path is not None:
            self.dataset_path_gt_UDC_toled = os.path.join(UDC_toled_path, 'HQ_256')
            self.dataset_path_lq_UDC_toled = os.path.join(UDC_toled_path, 'LQ_256')
            self.paths_gt_UDC_toled, self.sizes_gt_UDC_toled = util.get_image_paths('img', self.dataset_path_gt_UDC_toled)
            self.paths_lq_UDC_toled, self.sizes_lq_UDC_toled = util.get_image_paths('img', self.dataset_path_lq_UDC_toled)
            self.paths_UDC_toled_len = len(self.paths_lq_UDC_toled)
            sorted(self.paths_gt_UDC_toled)
            sorted(self.paths_lq_UDC_toled)
        
        # data/Enhancement/WhiteBalance
        if WhiteBalance_path is not None:
            # Multiple lq images to one gt image
            self.dataset_path_gt_WhiteBalance = os.path.join(WhiteBalance_path, 'Set1_ground_truth_images')
            self.dataset_path_lq_WhiteBalance = os.path.join(WhiteBalance_path, 'Set1_input_images_JPG')
            self.paths_gt_WhiteBalance, self.sizes_gt_WhiteBalance = util.get_image_paths('img', self.dataset_path_gt_WhiteBalance)
            self.paths_lq_WhiteBalance, self.sizes_lq_WhiteBalance = util.get_image_paths('img', self.dataset_path_lq_WhiteBalance)
            self.paths_lq_WhiteBalance_len = len(self.paths_lq_WhiteBalance)
            self.paths_gt_WhiteBalance_len = len(self.paths_gt_WhiteBalance)
            sorted(self.paths_gt_WhiteBalance)
            sorted(self.paths_lq_WhiteBalance)
        
        # data/Enhancement/UIEB_dive
        # UIEB dataset (UIEBD - Color Correction)
        if Color_Corre_path is not None:
            self.dataset_path_gt_Color_Corre = os.path.join(Color_Corre_path, 'dive_resize/train')
            self.dataset_path_lq_Color_Corre = os.path.join(Color_Corre_path, 'raw_resize/train')
            self.paths_gt_Color_Corre, self.sizes_gt_Color_Corre = util.get_image_paths('img', self.dataset_path_gt_Color_Corre)
            self.paths_lq_Color_Corre, self.sizes_lq_Color_Corre = util.get_image_paths('img', self.dataset_path_lq_Color_Corre)
            self.paths_Color_Corre_len = len(self.paths_lq_Color_Corre)
            sorted(self.paths_gt_Color_Corre)
            sorted(self.paths_lq_Color_Corre)
            
        # vignetting removal dataset
        # data/Enhancement/vignetting_removal512
        if VignettingRemoval_path is not None:
            self.dataset_path_gt_VignettingRemoval = os.path.join(VignettingRemoval_path, 'gt')
            self.dataset_path_lq_VignettingRemoval = os.path.join(VignettingRemoval_path, 'lq')
            self.paths_gt_VignettingRemoval, self.sizes_gt_VignettingRemoval = util.get_image_paths('img', self.dataset_path_gt_VignettingRemoval)
            self.paths_lq_VignettingRemoval, self.sizes_lq_VignettingRemoval = util.get_image_paths('img', self.dataset_path_lq_VignettingRemoval)
            self.paths_VignettingRemoval_len = len(self.paths_lq_VignettingRemoval)
            sorted(self.paths_gt_VignettingRemoval)
            sorted(self.paths_lq_VignettingRemoval)
        
        # data/Enhancement/MIT-fivek
        # MultiScale Tone Manipulation dataset (MIT-Adobe FiveK - MultiTone)
        if MultiTone_path is not None:
            self.dataset_path_gt_MultiTone = os.path.join(MultiTone_path, 'expert_C_Multiscale_tonemapping_matlab_train')
            self.dataset_path_lq_MultiTone = os.path.join(MultiTone_path, 'expert_C_train')
            self.paths_gt_MultiTone, self.sizes_gt_MultiTone = util.get_image_paths('img', self.dataset_path_gt_MultiTone)
            self.paths_lq_MultiTone, self.sizes_lq_MultiTone = util.get_image_paths('img', self.dataset_path_lq_MultiTone)
            assert len(self.paths_gt_MultiTone) == len(self.paths_lq_MultiTone), 'Error: MultiTone dataset length error. length of gt: {}, length of lq: {}'.format(len(self.paths_gt_MultiTone), len(self.paths_lq_MultiTone))
            self.paths_MultiTone_len = len(self.paths_lq_MultiTone)
            sorted(self.paths_gt_MultiTone)
            sorted(self.paths_lq_MultiTone)
        
        # data/Enhancement/ISP
        if ISP_path is not None:
            self.dataset_path_gt_ISP = os.path.join(ISP_path, 'gt')
            self.dataset_path_lq_ISP = os.path.join(ISP_path, 'lq')
            self.paths_gt_ISP, self.sizes_gt_ISP = util.get_image_paths('img', self.dataset_path_gt_ISP)
            self.paths_lq_ISP, self.sizes_lq_ISP = util.get_image_paths('img', self.dataset_path_lq_ISP)
            self.paths_ISP_len = len(self.paths_lq_ISP)
            sorted(self.paths_gt_ISP)
            sorted(self.paths_lq_ISP)
        
        ##########################################################################
        # Image Translation
        
        if DepthEstimation_path is not None:
            self.dataset_path_gt_DepthEstimation = os.path.join(DepthEstimation_path, 'output')
            self.dataset_path_lq_DepthEstimation = os.path.join(DepthEstimation_path, 'input')
            self.paths_gt_DepthEstimation, self.sizes_gt_DepthEstimation = util.get_image_paths('img', self.dataset_path_gt_DepthEstimation)
            self.paths_lq_DepthEstimation, self.sizes_lq_DepthEstimation = util.get_image_paths('img', self.dataset_path_lq_DepthEstimation)
            self.paths_DepthEstimation_len = len(self.paths_lq_DepthEstimation)
            sorted(self.paths_gt_DepthEstimation)
            sorted(self.paths_lq_DepthEstimation)
        
        # PencilDrawing dataset (MIT-Adobe FiveK - PencilDrawing by Combining sketch and tone for pencil drawing production)
        # data/Enhancement/MIT-fivek
        if PencilDrawing_path is not None:
            self.dataset_path_gt_PencilDrawing = os.path.join(PencilDrawing_path, 'expert_C_pencil_matlab_train')
            self.dataset_path_lq_PencilDrawing = os.path.join(PencilDrawing_path, 'expert_C_train')
            self.paths_gt_PencilDrawing, self.sizes_gt_PencilDrawing = util.get_image_paths('img', self.dataset_path_gt_PencilDrawing)
            self.paths_lq_PencilDrawing, self.sizes_lq_PencilDrawing = util.get_image_paths('img', self.dataset_path_lq_PencilDrawing)
            assert len(self.paths_gt_PencilDrawing) == len(self.paths_lq_PencilDrawing), 'Error: PencilDrawing dataset length error. length of gt: {}, length of lq: {}'.format(len(self.paths_gt_PencilDrawing), len(self.paths_lq_PencilDrawing))
            self.paths_PencilDrawing_len = len(self.paths_lq_PencilDrawing)
            sorted(self.paths_gt_PencilDrawing)
            sorted(self.paths_lq_PencilDrawing)
        
        # Photographic Style dataset (MIT-Adobe FiveK)
        if Photographic_path is not None:
            self.dataset_path_gt_Photographic = os.path.join(Photographic_path, 'expert_C_photographic_style_matlab_train')
            self.dataset_path_lq_Photographic = os.path.join(Photographic_path, 'expert_C_train')
            self.paths_gt_Photographic, self.sizes_gt_Photographic = util.get_image_paths('img', self.dataset_path_gt_Photographic)
            self.paths_lq_Photographic, self.sizes_lq_Photographic = util.get_image_paths('img', self.dataset_path_lq_Photographic)
            self.paths_Photographic_len = len(self.paths_lq_Photographic)
            sorted(self.paths_gt_Photographic)
            sorted(self.paths_lq_Photographic)
            
        # Relative total variation dataset (MIT-Adobe FiveK)
        if RTV_path is not None:
            self.dataset_path_gt_RTV = os.path.join(RTV_path, 'expert_C_relative_total_variation_matlab_train')
            self.dataset_path_lq_RTV = os.path.join(RTV_path, 'expert_C_train')
            self.paths_gt_RTV, self.sizes_gt_RTV = util.get_image_paths('img', self.dataset_path_gt_RTV)
            self.paths_lq_RTV, self.sizes_lq_RTV = util.get_image_paths('img', self.dataset_path_lq_RTV)
            self.paths_RTV_len = len(self.paths_lq_RTV)
            sorted(self.paths_gt_RTV)
            sorted(self.paths_lq_RTV)
        
        # data/ImageTranslation/salient_object_detection/train
        if SaliencyObjectDetection_path is not None:
            self.dataset_path_gt_SaliencyObjectDetection = os.path.join(SaliencyObjectDetection_path, 'output')
            self.dataset_path_lq_SaliencyObjectDetection = os.path.join(SaliencyObjectDetection_path, 'input')
            self.paths_gt_SaliencyObjectDetection, self.sizes_gt_SaliencyObjectDetection = util.get_image_paths('img', self.dataset_path_gt_SaliencyObjectDetection)
            self.paths_lq_SaliencyObjectDetection, self.sizes_lq_SaliencyObjectDetection = util.get_image_paths('img', self.dataset_path_lq_SaliencyObjectDetection)
            self.paths_SaliencyObjectDetection_len = len(self.paths_lq_SaliencyObjectDetection)
            sorted(self.paths_gt_SaliencyObjectDetection)
            sorted(self.paths_lq_SaliencyObjectDetection)
        
        # data/Enhancement/MIT-fivek/expert_C_style_Cloisonnism_AdaAttN_train
        if Style_Cloisonnism_path is not None:
            self.dataset_path_gt_Style_Cloisonnism = os.path.join(Style_Cloisonnism_path, 'expert_C_style_Cloisonnism_AdaAttN_train')
            self.dataset_path_lq_Style_Cloisonnism = os.path.join(Style_Cloisonnism_path, 'expert_C_train')
            self.paths_gt_Style_Cloisonnism, self.sizes_gt_Style_Cloisonnism = util.get_image_paths('img', self.dataset_path_gt_Style_Cloisonnism)
            
            style1 = [p for p in self.paths_gt_Style_Cloisonnism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Cloisonnism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Cloisonnism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Cloisonnism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Cloisonnism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Cloisonnism_singlelist_len = len(style1)
            self.lists_Style_Cloisonnism = [style1, style2, style3, style4, style5]
        
        # data/Enhancement/MIT-fivek/expert_C_style_Divisionism_AdaAttN_train
        if Style_Divisionism_path is not None:
            self.dataset_path_gt_Style_Divisionism = os.path.join(Style_Divisionism_path, 'expert_C_style_Divisionism_AdaAttN_train')
            self.dataset_path_lq_Style_Divisionism = os.path.join(Style_Divisionism_path, 'expert_C_train')
            self.paths_gt_Style_Divisionism, self.sizes_gt_Style_Divisionism = util.get_image_paths('img', self.dataset_path_gt_Style_Divisionism)
            
            style1 = [p for p in self.paths_gt_Style_Divisionism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Divisionism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Divisionism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Divisionism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Divisionism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Divisionism_singlelist_len = len(style1)
            self.lists_Style_Divisionism = [style1, style2, style3, style4, style5]
        
        # data/Enhancement/MIT-fivek/expert_C_style_Fauvism_AdaAttN_train
        if Style_Fauvism_path is not None:
            self.dataset_path_gt_Style_Fauvism = os.path.join(Style_Fauvism_path, 'expert_C_style_Fauvism_AdaAttN_train')
            self.dataset_path_lq_Style_Fauvism = os.path.join(Style_Fauvism_path, 'expert_C_train')
            self.paths_gt_Style_Fauvism, self.sizes_gt_Style_Fauvism = util.get_image_paths('img', self.dataset_path_gt_Style_Fauvism)
            
            style1 = [p for p in self.paths_gt_Style_Fauvism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Fauvism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Fauvism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Fauvism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Fauvism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Fauvism_singlelist_len = len(style1)
            self.lists_Style_Fauvism = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_JOJO_AdaAttN_train
        if Style_JOJO_path is not None:
            self.dataset_path_gt_Style_JOJO = os.path.join(Style_JOJO_path, 'expert_C_style_JOJO_AdaAttN_train')
            self.dataset_path_lq_Style_JOJO = os.path.join(Style_JOJO_path, 'expert_C_train')
            self.paths_gt_Style_JOJO, self.sizes_gt_Style_JOJO = util.get_image_paths('img', self.dataset_path_gt_Style_JOJO)
            
            style1 = [p for p in self.paths_gt_Style_JOJO if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_JOJO if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_JOJO if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_JOJO if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_JOJO if os.path.basename(p)[:2]=='5_']
            
            self.paths_JOJO_singlelist_len = len(style1)
            self.lists_Style_JOJO = [style1, style2, style3, style4, style4]

        # data/Enhancement/MIT-fivek/expert_C_Style_Johannes_Vermeer_AdaAttN_train
        if Style_Vermeer_path is not None:
            self.dataset_path_gt_Style_Vermeer = os.path.join(Style_Vermeer_path, 'expert_C_Style_Johannes_Vermeer_AdaAttN_train')
            self.dataset_path_lq_Style_Vermeer = os.path.join(Style_Vermeer_path, 'expert_C_train')
            self.paths_gt_Style_Vermeer, self.sizes_gt_Style_Vermeer = util.get_image_paths('img', self.dataset_path_gt_Style_Vermeer)
            
            style1 = [p for p in self.paths_gt_Style_Vermeer if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Vermeer if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Vermeer if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Vermeer if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Vermeer if os.path.basename(p)[:2]=='5_']
            
            self.paths_Vermeer_singlelist_len = len(style1)
            self.lists_Style_Vermeer = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_Raphael_AdaAttN_train
        if Style_Raphael_path is not None:
            self.dataset_path_gt_Style_Raphael = os.path.join(Style_Raphael_path, 'expert_C_style_Raphael_AdaAttN_train')
            self.dataset_path_lq_Style_Raphael = os.path.join(Style_Raphael_path, 'expert_C_train')
            self.paths_gt_Style_Raphael, self.sizes_gt_Style_Raphael = util.get_image_paths('img', self.dataset_path_gt_Style_Raphael)
            
            style1 = [p for p in self.paths_gt_Style_Raphael if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Raphael if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Raphael if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Raphael if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Raphael if os.path.basename(p)[:2]=='5_']
            
            self.paths_Raphael_singlelist_len = len(style1)
            self.lists_Style_Raphael = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_modernism_AdaAttN_train
        if Style_Modernism_path is not None:
            self.dataset_path_gt_Style_Modernism = os.path.join(Style_Modernism_path, 'expert_C_style_modernism_AdaAttN_train')
            self.dataset_path_lq_Style_Modernism = os.path.join(Style_Modernism_path, 'expert_C_train')
            self.paths_gt_Style_Modernism, self.sizes_gt_Style_Modernism = util.get_image_paths('img', self.dataset_path_gt_Style_Modernism)
            
            style1 = [p for p in self.paths_gt_Style_Modernism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Modernism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Modernism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Modernism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Modernism if os.path.basename(p)[:2]=='5_']
            
            self.paths_Modernism_singlelist_len = len(style1)
            self.lists_Style_Modernism = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_monet_AdaAttN_train
        if Style_Monet_path is not None:
            self.dataset_path_gt_Style_Monet = os.path.join(Style_Monet_path, 'expert_C_style_monet_AdaAttN_train')
            self.dataset_path_lq_Style_Monet = os.path.join(Style_Monet_path, 'expert_C_train')
            self.paths_gt_Style_Monet, self.sizes_gt_Style_Monet = util.get_image_paths('img', self.dataset_path_gt_Style_Monet)
            
            style1 = [p for p in self.paths_gt_Style_Monet if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Monet if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Monet if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Monet if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Monet if os.path.basename(p)[:2]=='5_']
            
            self.paths_Monet_singlelist_len = len(style1)
            self.lists_Style_Monet = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_neo_impressionism_AdaAttN_train
        if Style_NeoImpressionism_path is not None:
            self.dataset_path_gt_Style_NeoImpressionism = os.path.join(Style_NeoImpressionism_path, 'expert_C_style_neo_impressionism_AdaAttN_train')
            self.dataset_path_lq_Style_NeoImpressionism = os.path.join(Style_NeoImpressionism_path, 'expert_C_train')
            self.paths_gt_Style_NeoImpressionism, self.sizes_gt_Style_NeoImpressionism = util.get_image_paths('img', self.dataset_path_gt_Style_NeoImpressionism)
            
            style1 = [p for p in self.paths_gt_Style_NeoImpressionism if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_NeoImpressionism if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_NeoImpressionism if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_NeoImpressionism if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_NeoImpressionism if os.path.basename(p)[:2]=='5_']
            
            self.paths_NeoImpressionism_singlelist_len = len(style1)
            self.lists_Style_NeoImpressionism = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_pop_art_AdaAttN_train
        if Style_PopArt_path is not None:
            self.dataset_path_gt_Style_PopArt = os.path.join(Style_PopArt_path, 'expert_C_style_pop_art_AdaAttN_train')
            self.dataset_path_lq_Style_PopArt = os.path.join(Style_PopArt_path, 'expert_C_train')
            self.paths_gt_Style_PopArt, self.sizes_gt_Style_PopArt = util.get_image_paths('img', self.dataset_path_gt_Style_PopArt)
            
            style1 = [p for p in self.paths_gt_Style_PopArt if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_PopArt if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_PopArt if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_PopArt if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_PopArt if os.path.basename(p)[:2]=='5_']
            
            self.paths_PopArt_singlelist_len = len(style1)
            self.lists_Style_PopArt = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_ukiyoe_AdaAttN_train
        if Style_Ukiyoe_path is not None:
            self.dataset_path_gt_Style_Ukiyoe = os.path.join(Style_Ukiyoe_path, 'expert_C_style_ukiyoe_AdaAttN_train')
            self.dataset_path_lq_Style_Ukiyoe = os.path.join(Style_Ukiyoe_path, 'expert_C_train')
            self.paths_gt_Style_Ukiyoe, self.sizes_gt_Style_Ukiyoe = util.get_image_paths('img', self.dataset_path_gt_Style_Ukiyoe)
            
            style1 = [p for p in self.paths_gt_Style_Ukiyoe if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_Ukiyoe if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_Ukiyoe if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_Ukiyoe if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_Ukiyoe if os.path.basename(p)[:2]=='5_']
            
            self.paths_Ukiyoe_singlelist_len = len(style1)
            self.lists_Style_Ukiyoe = [style1, style2, style3, style4, style4]
        
        # data/Enhancement/MIT-fivek/expert_C_style_vangogh_AdaAttN_train
        if Style_VanGogh_path is not None:
            self.dataset_path_gt_Style_VanGogh = os.path.join(Style_VanGogh_path, 'expert_C_style_vangogh_AdaAttN_train')
            self.dataset_path_lq_Style_VanGogh = os.path.join(Style_VanGogh_path, 'expert_C_train')
            self.paths_gt_Style_VanGogh, self.sizes_gt_Style_VanGogh = util.get_image_paths('img', self.dataset_path_gt_Style_VanGogh)
            
            style1 = [p for p in self.paths_gt_Style_VanGogh if os.path.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_gt_Style_VanGogh if os.path.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_gt_Style_VanGogh if os.path.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_gt_Style_VanGogh if os.path.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_gt_Style_VanGogh if os.path.basename(p)[:2]=='5_']
            
            self.paths_VanGogh_singlelist_len = len(style1)
            self.lists_Style_VanGogh = [style1, style2, style3, style4, style4]
        
        ##########################################################################
        # Edge Detection
        # Edge Detection dataset (BIPED)
        if Edge_Detect_path is not None:
            self.dataset_path_gt_Edge_Detect = os.path.join(Edge_Detect_path, 'edges/train')
            self.dataset_path_lq_Edge_Detect = os.path.join(Edge_Detect_path, 'imgs/train')
            self.paths_gt_Edge_Detect, self.sizes_gt_Edge_Detect = util.get_image_paths('img', self.dataset_path_gt_Edge_Detect)
            self.paths_lq_Edge_Detect, self.sizes_lq_Edge_Detect = util.get_image_paths('img', self.dataset_path_lq_Edge_Detect)
            self.paths_Edge_Detect_len = len(self.paths_lq_Edge_Detect)
            sorted(self.paths_gt_Edge_Detect)
            sorted(self.paths_lq_Edge_Detect)
        
        ##########################################################################
        # dataset list   
        if self.tasks_version == 0:
            self.dataset_list = ['Base', 'ITS', 'Demoireing', 'Rain13K', 'DustRemoval',
                                 'Desnowing', 'Base', 'FaceSR', 'HighlightRemoval', # FlareRemoval has some problem
                                 'LOL', 'ReflectionRemoval', 'ShadowRemoval_ISTD', 'FiveK',
                                 'ShadowRemoval_ISTD_adjusted', 'ShadowRemoval_SRD', 'LLF', 
                                 'TextSR_train1', 'TextSR_train2', 'Backlit', 'HistoEqu', 
                                 'ExposureError', 'InsFilterRemoval', 'InsFilterAddition',
                                 'Bokeh', 'UDC_poled', 'UDC_toled', 'ColorCorre', 'MultiTone', 
                                 'SDRHDR', 'Base', 'EdgeDetec', 'PencialDraw', 'RTV', # WhiteBalance has some problem
                                 'Photographic', 'styleClo', 'styleDiv', 'styleFau', 'styleRaph',
                                 'styleVermeer', 'styleJOJO', 'styleModern', 'styleMonet',
                                 'styleNeo', 'stylePopArt', 'styleUkiyoe', 'styleVanGogh'] # 46
            # single task
            self.onthefly_degradation_list1 = ['LowLight', 'Rain', 'Ringing', 'r_l', 
                                               'Inpainting', 'Defocus_Blur', 'mosaic']
            self.onthefly_degradation_list2 = ['blur', 'noise', 'compression', 'brighten', 'darken',
                                      'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen',
                                      'saturate_weaken', 'oversharpen', 'pixelate', 'quantization']
            self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        
        elif self.tasks_version == 1:
            self.dataset_list = ['CT_Covid_SR', 'CT_Covid_GaussianNoise', 'ITS', 'Rain13K', 'Demoireing', 'DustRemoval',
                                 'Desnowing', 'FlareRemoval', 'FaceSR', 'HighlightRemoval', 'LOL', 'MRI_SR', # 'LensFlare', 有些时候完全看不到distortion
                                 'MRI_Denoise', 'MarineSnowRemovalSmall', 'MarineSnowRemovalVarious', 'ReflectionRemoval', 'ShadowRemoval_ISTD', 'ShadowRemoval_ISTD_adjusted',
                                 'ShadowRemoval_SRD', 'SatelliteSR', 'SatelliteGaussianNoise', 'TextSR_train1', 'TextSR_train2',
                                 'ThinCloudRemoval', 'Watermark', 'Backlit', 'DSLR_blackberry', 'DSLR_iphone', 'DSLR_sony', 'RealLowLightSR',
                                 'ExposureError', 'InsFilterRemoval', 'InsFilterAddition', 'ISP', 'FiveK', 'LLF', 'Bokeh', 'UDC_poled', 'UDC_toled',
                                 'HistoEqu', 'ColorCorre', 'MultiTone', 'SDRHDR', 'VignettingRemoval',  'EdgeDetec', 'PencialDraw', 'DepthEstimation', 'WhiteBalance',
                                 'Photographic', 'RTV', 'SaliencyObjectDetection', 'styleClo', 'styleDiv', 'styleFau', 'styleVermeer', 'styleJOJO',
                                 'styleRaph', 'styleModern', 'styleMonet', 'styleNeo', 'stylePopArt', 'styleUkiyoe', 'styleVanGogh'] # 63
            # single task
            self.onthefly_degradation_list1 = ['LowLight', 'Rain', 'Ringing', 'r_l', 
                                               'Inpainting', 'Defocus_Blur', 'mosaic']
            self.onthefly_degradation_list2 = ['blur', 'noise', 'compression', 'brighten', 'darken',
                                      'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen',
                                      'saturate_weaken', 'oversharpen', 'pixelate', 'quantization']
            self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        
        elif self.tasks_version == 2:
            self.dataset_list=['ReflectionRemoval']
        else:
            print('Wrong task flag!')
            sys.exit(1)
            
        print('dataset list: ', self.dataset_list)
        # print('degradation_type list: ', self.degradation_type_list)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        if self.tasks_version == 0:
            dataset_choice = np.random.choice(self.dataset_list, p=[16/64, 1/64, 1/64, 1/64, 1/64, 1/64, 1/64,
                                                                    1/64, 1/64, 1/64, 1/64, 1/64, 1/64, 1/64, 
                                                                    1/64, 1/64, 1/64, 1/64, 1/64, 1/64, 1/64,
                                                                    2/64, 2/64, 1/64, 1/64, 1/64, 1/64, 1/64,
                                                                    2/64, 1/64, 1/64, 1/64, 1/64, 1/64, 1/64,
                                                                    1/64, 1/64, 1/64, 1/64, 1/64, 1/64, 1/64,
                                                                    1/64, 1/64, 1/64, 1/64])
        else:
            dataset_choice = self.dataset_list[idx%len(self.dataset_list)]
        if dataset_choice == 'Base':
            tasks_list = ['GenLV-Degradation-plus', 'X-Distortion', 'Operators']
            
            random_index1 = random.randint(0, self.paths_base_len-1)
            gt1_path = self.paths_gt[random_index1]
            random_index2 = random.randint(0, self.paths_base_len-1)
            gt2_path = self.paths_gt[random_index2]
            
            img_gt1 = util.read_img(None, gt1_path, None, float=False) # np.uint8
            img_gt2 = util.read_img(None, gt2_path, None, float=False) # np.uint8

            # if the image size is too small
            # H, W, _ = img_gt1.shape
            # if H < self.gt_size or W < self.gt_size:
            #     img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)
                
            # H, W, _ = img_gt2.shape
            # if H < self.gt_size or W < self.gt_size:
            #     img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)
            
            # if the image size is not same as self.gt_size
            H, W, _ = img_gt1.shape
            if H != self.gt_size or W != self.gt_size:
                img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_AREA)
                
            H, W, _ = img_gt2.shape
            if H != self.gt_size or W != self.gt_size:
                img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_AREA)
            
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
            
            # add on-the-fly degradation
            tasks_flag = np.random.choice(tasks_list, p=[1/2, 7/16, 1/16])
            if tasks_flag == 'GenLV-Degradation-plus':
                img_gt1 = uint2single(img_gt1)
                img_gt2 = uint2single(img_gt2)
                round_select = random.choice(['Single', 'Mix1R'])
                if round_select == 'Single':
                    deg_type = random.choice(self.onthefly_degradation_list1)
                    img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
                # 1 round Mix-degradation
                elif round_select == 'Mix1R':
                    degradation_type_1 = ['LowLight', 'None', 'None', 'None', 'None']
                    degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', 'None', 'None']
                    degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None', 'None']
                    degradation_type_4 = ['Inpainting', 'Rain', 'None', 'None', 'None']    
                    degradation_type_5 = ['JPEG', 'None', 'None', 'None', 'None']
                    
                    deg_type1 = random.choice(degradation_type_1)
                    deg_type2 = random.choice(degradation_type_2)
                    deg_type3 = random.choice(degradation_type_3)
                    deg_type4 = random.choice(degradation_type_4)
                    deg_type5 = random.choice(degradation_type_5)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type1)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type2)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type3)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type4)
                    img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type5)
                    deg_type = 'Mix1R_' + deg_type1 + '_' + deg_type2 + '_' + deg_type3 + '_' + deg_type4 + '_' + deg_type5
                    
            elif tasks_flag == 'X-Distortion':
                deg_type = random.choice(self.onthefly_degradation_list2)
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
        
        elif dataset_choice == 'CT_Covid_SR':
            random_index1 = random.randint(0, self.paths_CT_Covid_len-1)
            gt1_path = self.paths_gt_CT_Covid[random_index1]
            
            random_index2 = random.randint(0, self.paths_CT_Covid_len-1)
            gt2_path = self.paths_gt_CT_Covid[random_index2]
            
            deg_type = 'CT_Covid_SR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            
            img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), 'Resize')
        
        elif dataset_choice == 'CT_Covid_GaussianNoise':
            random_index1 = random.randint(0, self.paths_CT_Covid_len-1)
            gt1_path = self.paths_gt_CT_Covid[random_index1]
            
            random_index2 = random.randint(0, self.paths_CT_Covid_len-1)
            gt2_path = self.paths_gt_CT_Covid[random_index2]
            
            deg_type = 'CT_Covid_GaussianNoise'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            
            img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), 'GaussianNoise')
            
        elif dataset_choice == 'ITS':
            random_index1 = random.randint(0, self.paths_ITS_len-1)
            lq1_path = self.paths_lq_ITS[random_index1]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            gt1_path = os.path.join(self.dataset_path_gt_ITS, '{}.png'.format(gt1_name))
    
            random_index2 = random.randint(0, self.paths_ITS_len-1)
            lq2_path = self.paths_lq_ITS[random_index2]
            gt2_name = lq2_path.split('/')[-1].split('_')[0]
            gt2_path = os.path.join(self.dataset_path_gt_ITS, '{}.png'.format(gt2_name))

            deg_type = 'ITS'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'Rain13K':
            random_index1 = random.randint(0, self.paths_Rain13K_len-1)
            lq1_path = self.paths_lq_Rain13K[random_index1]
            gt1_path = self.paths_gt_Rain13K[random_index1]
    
            random_index2 = random.randint(0, self.paths_Rain13K_len-1)
            lq2_path = self.paths_lq_Rain13K[random_index2]
            gt2_path = self.paths_gt_Rain13K[random_index2]
            
            deg_type = 'Rain13K'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
            # H, W, _ = img_gt1.shape
            # if H < self.gt_size or W < self.gt_size:
            #     img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)
            # H, W, _ = img_gt2.shape
            # if H < self.gt_size or W < self.gt_size:
            #     img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)              
            # H, W, _ = img_lq1.shape
            # #print(H, W)
            # if H < self.gt_size or W < self.gt_size:
            #     img_lq1 = cv2.resize(img_lq1, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)
            # H, W, _ = img_lq2.shape
            # if H < self.gt_size or W < self.gt_size:
            #     img_lq2 = cv2.resize(img_lq2, (self.gt_size, self.gt_size),
            #                         interpolation=cv2.INTER_LINEAR)

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

        elif dataset_choice == 'Desnowing':
            random_index1 = random.randint(0, self.paths_Desnowing_len-1)
            lq1_path = self.paths_lq_Desnowing[random_index1]
            gt1_path = self.paths_gt_Desnowing[random_index1]
    
            random_index2 = random.randint(0, self.paths_Desnowing_len-1)
            lq2_path = self.paths_lq_Desnowing[random_index2]
            gt2_path = self.paths_gt_Desnowing[random_index2]
            
            deg_type = 'Desnowing'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # need further check: why one-the-fly
        elif dataset_choice == 'FlareRemoval':
            random_index1 = random.randint(0, self.paths_FlareRemoval_len-1)
            gt1_path = self.paths_gt_FlareRemoval[random_index1]
    
            random_index2 = random.randint(0, self.paths_FlareRemoval_len-1)
            gt2_path = self.paths_gt_FlareRemoval[random_index2]
            
            deg_type = 'FlareRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            
            # load falre image
            flare_list = ['Synthetic', 'Real']
            reflective_list = ['None', 'Reflective']
            light_source_list = ['Synthetic', 'Real', 'None']
            
            flare_type = random.choice(flare_list)
            reflective_type = random.choice(reflective_list)
            light_source_type = random.choice(light_source_list)
            light_source_type = 'None' if flare_type == 'None' else flare_type
            
            if flare_type == 'Synthetic':
                flare_path = self.paths_scattering_synthesis[random.randint(0, len(self.paths_scattering_synthesis)-1)]
                flare_name = os.path.basename(flare_path).split('.')[0]
            else:
                flare_path = self.paths_scattering_real[random.randint(0, len(self.paths_scattering_real)-1)]
                flare_name = os.path.basename(flare_path).split('.')[0]
                reflective_type = 'None'
            
            if reflective_type == 'Reflective':
                reflective_path = self.paths_reflective[random.randint(0, len(self.paths_reflective)-1)]
            else:
                reflective_path = None
                
            if light_source_type == 'Synthetic':
                light_source_path = os.path.join(self.dataset_path_light_synthesis, flare_name + '.png')
            elif light_source_type == 'Real':
                light_source_path = os.path.join(self.dataset_path_light_real, flare_name + '.png')
            else:
                light_source_path = None
                
            img_flare = util.read_img(None, flare_path, None)
            img_relaticve = util.read_img(None, reflective_path, None) if reflective_path is not None else None
            img_light_source = util.read_img(None, light_source_path, None) if light_source_path is not None else None
            
            img_lq1, img_gt1 = add_flare(img_gt1.copy(), img_flare, img_relaticve, img_light_source)
            img_lq2, img_gt2 = add_flare(img_gt2.copy(), img_flare, img_relaticve, img_light_source)
            
        elif dataset_choice == 'FaceSR':
            random_index1 = random.randint(0, self.paths_Desnowing_len-1)
            gt1_path = self.paths_gt_FaceSR[random_index1]
    
            random_index2 = random.randint(0, self.paths_Desnowing_len-1)
            gt2_path = self.paths_gt_FaceSR[random_index2]
            
            deg_type = 'FaceSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)

            img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1.copy(), img_gt2.copy(), deg_type='Resize')
            
        elif dataset_choice == 'HighlightRemoval':
            random_index1 = random.randint(0, self.paths_LOL_len-1)
            lq1_path = self.paths_lq_HighlightRemoval[random_index1]
            gt1_path = self.paths_gt_HighlightRemoval[random_index1]
            
            random_index2 = random.randint(0, self.paths_LOL_len-1)
            lq2_path = self.paths_lq_HighlightRemoval[random_index2]
            gt2_path = self.paths_gt_HighlightRemoval[random_index2]
            
            deg_type = 'HighlightRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'LensFlare':
            random_index1 = random.randint(0, self.paths_LensFlare_len-1)
            gt1_path = self.paths_gt_LensFlare[random_index1]
            
            random_index2 = random.randint(0, self.paths_LensFlare_len-1)
            gt2_path = self.paths_gt_LensFlare[random_index2]
            
            deg_type = 'LensFlare'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            
            flare_list = ['Synthetic', 'Real']
            flare_type = random.choice(flare_list)
            
            if flare_type == 'Synthetic':
                flare_path = self.paths_flare_simulated[random.randint(0, len(self.paths_flare_simulated)-1)]
            else:
                flare_path = self.paths_flare_captured[random.randint(0, len(self.paths_flare_captured)-1)]
                
            img_flare = util.read_img(None, flare_path, None)
            
            img_lq1, img_gt1 = add_lens_flare(img_gt1.copy(), img_flare)
            img_lq2, img_gt2 = add_lens_flare(img_gt2.copy(), img_flare)
            
        elif dataset_choice == 'LOL':
            random_index1 = random.randint(0, self.paths_LOL_len-1)
            lq1_path = self.paths_lq_LOL[random_index1]
            gt1_path = self.paths_gt_LOL[random_index1]
            
            random_index2 = random.randint(0, self.paths_LOL_len-1)
            lq2_path = self.paths_lq_LOL[random_index2]
            gt2_path = self.paths_gt_LOL[random_index2]
            
            deg_type = 'LOL'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'MRI_SR':
            random_index1 = random.randint(0, self.paths_MRI_len-1)
            gt1_path = self.paths_gt_MRI[random_index1]
            
            random_index2 = random.randint(0, self.paths_MRI_len-1)
            gt2_path = self.paths_gt_MRI[random_index2]
            
            deg_type = 'MRI_SR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1.copy(), img_gt2.copy(), deg_type='Resize')

        elif dataset_choice == 'MRI_Denoise':
            random_index1 = random.randint(0, self.paths_MRI_len-1)
            gt1_path = self.paths_gt_MRI[random_index1]
            
            random_index2 = random.randint(0, self.paths_MRI_len-1)
            gt2_path = self.paths_gt_MRI[random_index2]
            
            deg_type = 'MRI_Denoise'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1.copy(), img_gt2.copy(), deg_type='GaussianNoise')
        
        elif dataset_choice == 'MarineSnowRemovalSmall':
            random_index1 = random.randint(0, self.paths_MarineSnowRemovalSmall_len-1)
            lq1_path = self.paths_lq_MarineSnowRemovalSmall[random_index1]
            gt1_path = self.paths_gt_MarineSnowRemovalSmall[random_index1]
            
            random_index2 = random.randint(0, self.paths_MarineSnowRemovalSmall_len-1)
            lq2_path = self.paths_lq_MarineSnowRemovalSmall[random_index2]
            gt2_path = self.paths_gt_MarineSnowRemovalSmall[random_index2]
            
            deg_type = 'MarineSnowRemovalSmall'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'MarineSnowRemovalVarious':
            random_index1 = random.randint(0, self.paths_MarineSnowRemovalVarious_len-1)
            lq1_path = self.paths_lq_MarineSnowRemovalVarious[random_index1]
            gt1_path = self.paths_gt_MarineSnowRemovalVarious[random_index1]
            
            random_index2 = random.randint(0, self.paths_MarineSnowRemovalVarious_len-1)
            lq2_path = self.paths_lq_MarineSnowRemovalVarious[random_index2]
            gt2_path = self.paths_gt_MarineSnowRemovalVarious[random_index2]
            
            deg_type = 'MarineSnowRemovalVarious'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'ReflectionRemoval':
            random_index1 = random.randint(0, self.paths_reflection_ReflectionRemoval_len-1)
            random_index2 = random.randint(0, self.paths_transmission_ReflectionRemoval_len-1)
            reflection1_path = self.paths_reflection_ReflectionRemoval[random_index1]
            transmission1_path = self.paths_transmission_ReflectionRemoval[random_index2]
            
            random_index3 = random.randint(0, self.paths_reflection_ReflectionRemoval_len-1)
            random_index4 = random.randint(0, self.paths_transmission_ReflectionRemoval_len-1)
            reflection2_path = self.paths_reflection_ReflectionRemoval[random_index3]
            transmission2_path = self.paths_transmission_ReflectionRemoval[random_index4]
            
            deg_type = 'ReflectionRemoval'
            img_reflection1 = util.read_img(None, reflection1_path, None)
            img_gt1 = util.read_img(None, transmission1_path, None)
            img_reflection2 = util.read_img(None, reflection2_path, None)
            img_gt2 = util.read_img(None, transmission2_path, None)
            
            # resize to 256
            img_reflection1 = cv2.resize(np.copy(img_reflection1), (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            img_gt1 = cv2.resize(np.copy(img_gt1), (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            img_reflection2 = cv2.resize(np.copy(img_reflection2), (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            img_gt2 = cv2.resize(np.copy(img_gt2), (self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            
            #generate lq
            _, _, img_lq1 = add_reflection(img_gt1.copy(), img_reflection1)
            _, _, img_lq2 = add_reflection(img_gt2.copy(), img_reflection2)
        
        elif dataset_choice == 'ShadowRemoval_ISTD':
            random_index1 = random.randint(0, self.paths_ShadowRemoval_ISTD_len-1)
            lq1_path = self.paths_lq_ShadowRemoval_ISTD[random_index1]
            gt1_path = self.paths_gt_ShadowRemoval_ISTD[random_index1]
            
            random_index2 = random.randint(0, self.paths_ShadowRemoval_ISTD_len-1)
            lq2_path = self.paths_lq_ShadowRemoval_ISTD[random_index2]
            gt2_path = self.paths_gt_ShadowRemoval_ISTD[random_index2]
            
            deg_type = 'ShadowRemoval_ISTD'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'ShadowRemoval_ISTD_adjusted':
            random_index1 = random.randint(0, self.paths_ShadowRemoval_ISTD_adjusted_len-1)
            lq1_path = self.paths_lq_ShadowRemoval_ISTD_adjusted[random_index1]
            gt1_path = self.paths_gt_ShadowRemoval_ISTD_adjusted[random_index1]
            
            random_index2 = random.randint(0, self.paths_ShadowRemoval_ISTD_adjusted_len-1)
            lq2_path = self.paths_lq_ShadowRemoval_ISTD_adjusted[random_index2]
            gt2_path = self.paths_gt_ShadowRemoval_ISTD_adjusted[random_index2]
            
            deg_type = 'ShadowRemoval_ISTD'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'ShadowRemoval_SRD':
            random_index1 = random.randint(0, self.paths_ShadowRemoval_SRD_len-1)
            lq1_path = self.paths_lq_ShadowRemoval_SRD[random_index1]
            gt1_path = self.paths_gt_ShadowRemoval_SRD[random_index1]
            
            random_index2 = random.randint(0, self.paths_ShadowRemoval_SRD_len-1)
            lq2_path = self.paths_lq_ShadowRemoval_SRD[random_index2]
            gt2_path = self.paths_gt_ShadowRemoval_SRD[random_index2]
            
            deg_type = 'ShadowRemoval_SRD'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'SatelliteSR':
            random_index1 = random.randint(0, self.paths_Satellite_len-1)
            gt1_path = self.paths_gt_Satellite[random_index1]
            
            random_index2 = random.randint(0, self.paths_Satellite_len-1)
            gt2_path = self.paths_gt_Satellite[random_index2]
            
            deg_type = 'SatelliteSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)

            img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1.copy(), img_gt2.copy(), deg_type='Resize')
        
        elif dataset_choice == 'SatelliteGaussianNoise':
            random_index1 = random.randint(0, self.paths_Satellite_len-1)
            gt1_path = self.paths_gt_Satellite[random_index1]
            
            random_index2 = random.randint(0, self.paths_Satellite_len-1)
            gt2_path = self.paths_gt_Satellite[random_index2]
            
            deg_type = 'SatelliteGaussianNoise'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)

            img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1.copy(), img_gt2.copy(), deg_type='GaussianNoise')
            
        elif dataset_choice == 'TextSR_train1':
            random_index1 = random.randint(0, self.paths_TextSR_train1_len-1)
            lq1_path = self.paths_lq_TextSR_train1[random_index1]
            gt1_path = self.paths_gt_TextSR_train1[random_index1]
            
            random_index2 = random.randint(0, self.paths_TextSR_train1_len-1)
            lq2_path = self.paths_lq_TextSR_train1[random_index2]
            gt2_path = self.paths_gt_TextSR_train1[random_index2]
            
            deg_type = 'TextSR_train1'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'TextSR_train2':
            random_index1 = random.randint(0, self.paths_TextSR_train2_len-1)
            lq1_path = self.paths_lq_TextSR_train2[random_index1]
            gt1_path = self.paths_gt_TextSR_train2[random_index1]
            
            random_index2 = random.randint(0, self.paths_TextSR_train2_len-1)
            lq2_path = self.paths_lq_TextSR_train2[random_index2]
            gt2_path = self.paths_gt_TextSR_train2[random_index2]
            
            deg_type = 'TextSR_train2'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'ThinCloudRemoval':
            random_index1 = random.randint(0, self.paths_ThinCloudRemoval_len-1)
            lq1_path = self.paths_lq_ThinCloudRemoval[random_index1]
            gt1_path = self.paths_gt_ThinCloudRemoval[random_index1]
            
            random_index2 = random.randint(0, self.paths_ThinCloudRemoval_len-1)
            lq2_path = self.paths_lq_ThinCloudRemoval[random_index2]
            gt2_path = self.paths_gt_ThinCloudRemoval[random_index2]
            
            deg_type = 'ThinCloudRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'Watermark':
            random_index1 = random.randint(0, self.paths_Watermark_len-1)
            lq1_path = self.paths_lq_Watermark[random_index1]
            gt1_path = self.paths_gt_Watermark[random_index1]
            
            random_index2 = random.randint(0, self.paths_Watermark_len-1)
            lq2_path = self.paths_lq_Watermark[random_index2]
            gt2_path = self.paths_gt_Watermark[random_index2]
            
            deg_type = 'Watermark'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'RealLowLightSR':
            random_index1 = random.randint(0, self.paths_RealLowLightSR_len-1)
            lq1_path = self.paths_lq_RealLowLightSR[random_index1]
            gt1_name = lq1_path.split('/')[-1].split('-')[0]
            gt1_path = os.path.join(self.dataset_path_gt_RealLowLightSR, '{}.png'.format(gt1_name))
            
            random_index2 = random.randint(0, self.paths_RealLowLightSR_len-1)
            lq2_path = self.paths_lq_RealLowLightSR[random_index2]
            gt2_name = lq2_path.split('/')[-1].split('-')[0]
            gt2_path = os.path.join(self.dataset_path_gt_RealLowLightSR, '{}.png'.format(gt2_name))
            
            deg_type = 'RealLowLightSR'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        ### Enhancement
        elif dataset_choice == 'Backlit':
            random_index1 = random.randint(0, self.paths_Backlit_len-1)
            lq1_path = self.paths_lq_Backlit[random_index1]
            gt1_path = self.paths_gt_Backlit[random_index1]
            
            random_index2 = random.randint(0, self.paths_Backlit_len-1)
            lq2_path = self.paths_lq_Backlit[random_index2]
            gt2_path = self.paths_gt_Backlit[random_index2]
            
            deg_type = 'Backlit'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'DSLR_blackberry':
            random_index1 = random.randint(0, self.paths_DSLR_blackberry_len-1)
            lq1_path = self.paths_lq_DSLR_blackberry[random_index1]
            gt1_path = self.paths_gt_DSLR_blackberry[random_index1]
            
            random_index2 = random.randint(0, self.paths_DSLR_blackberry_len-1)
            lq2_path = self.paths_lq_DSLR_blackberry[random_index2]
            gt2_path = self.paths_gt_DSLR_blackberry[random_index2]
            
            deg_type = 'DSLR_blackberry'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'DSLR_iphone':
            random_index1 = random.randint(0, self.paths_DSLR_iphone_len-1)
            lq1_path = self.paths_lq_DSLR_iphone[random_index1]
            gt1_path = self.paths_gt_DSLR_iphone[random_index1]
            
            random_index2 = random.randint(0, self.paths_DSLR_iphone_len-1)
            lq2_path = self.paths_lq_DSLR_iphone[random_index2]
            gt2_path = self.paths_gt_DSLR_iphone[random_index2]
            
            deg_type = 'DSLR_iphone'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'DSLR_sony':
            random_index1 = random.randint(0, self.paths_DSLR_iphone_len-1)
            lq1_path = self.paths_lq_DSLR_iphone[random_index1]
            gt1_path = self.paths_gt_DSLR_iphone[random_index1]
            
            random_index2 = random.randint(0, self.paths_DSLR_iphone_len-1)
            lq2_path = self.paths_lq_DSLR_iphone[random_index2]
            gt2_path = self.paths_gt_DSLR_iphone[random_index2]
            
            deg_type = 'DSLR_sony'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'ExposureError':
            # multiple lq to one gt
            random_index1 = random.randint(0, self.paths_lq_ExposureError_len-1)
            lq1_path = self.paths_lq_ExposureError[random_index1]
            gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-1])
            expo_suffix = lq1_path.split('_')[-1]
            gt1_path = os.path.join(self.dataset_path_gt_ExposureError, '{}.jpg'.format(gt1_name))
            
            random_index2 = random.randint(0, self.paths_gt_ExposureError_len-1)
            gt2_path = self.paths_gt_ExposureError[random_index2]
            lq2_name = os.path.splitext(gt2_path.split('/')[-1])[0]
            lq2_path = os.path.join(self.dataset_path_lq_ExposureError, '{}_{}'.format(lq2_name, expo_suffix))
            
            deg_type = 'ExposureError'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'InsFilterRemoval':
            # multiple to one
            random_index1 = random.randint(0, self.enh_paths_InstagramFilter_len-1)
            lq1_path = self.paths_enh_InstagramFilter[random_index1]
            filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            gt1_path = os.path.join(self.dataset_path_ori_InstagramFilter, '{}_Original.jpg'.format(gt1_name))
            
            random_index2 = random.randint(0, self.ori_paths_InstagramFilter_len-1)
            gt2_path = self.paths_ori_InstagramFilter[random_index2]
            lq2_name = gt2_path.split('/')[-1].split('_')[0]
            lq2_path = os.path.join(self.dataset_path_enh_InstagramFilter, '{}_{}'.format(lq2_name, filter_suffix))
            
            deg_type = 'InsFilterRemoval'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'InsFilterAddition':
            # multiple to one
            random_index1 = random.randint(0, self.enh_paths_InstagramFilter_len-1)
            lq1_path = self.paths_enh_InstagramFilter[random_index1]
            filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            gt1_path = os.path.join(self.dataset_path_ori_InstagramFilter, '{}_Original.jpg'.format(gt1_name))
            
            random_index2 = random.randint(0, self.ori_paths_InstagramFilter_len-1)
            gt2_path = self.paths_ori_InstagramFilter[random_index2]
            lq2_name = gt2_path.split('/')[-1].split('_')[0]
            lq2_path = os.path.join(self.dataset_path_enh_InstagramFilter, '{}_{}'.format(lq2_name, filter_suffix))
            
            deg_type = 'InsFilterAddition'
            img_gt1 = util.read_img(None, lq1_path, None)
            img_lq1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, lq2_path, None)
            img_lq2 = util.read_img(None, gt2_path, None)
        
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
            
        elif dataset_choice == 'FiveK':
            random_index1 = random.randint(0, self.paths_FiveK_len-1)
            lq1_path = self.paths_lq_FiveK[random_index1]
            gt1_path = self.paths_gt_FiveK[random_index1]
            
            random_index2 = random.randint(0, self.paths_FiveK_len-1)
            lq2_path = self.paths_lq_FiveK[random_index2]
            gt2_path = self.paths_gt_FiveK[random_index2]
            
            deg_type = 'FiveK'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)    
            
        elif dataset_choice == 'LLF':
            random_index1 = random.randint(0, self.paths_LLF_len-1)
            lq1_path = self.paths_lq_LLF[random_index1]
            gt1_path = self.paths_gt_LLF[random_index1]
            
            random_index2 = random.randint(0, self.paths_LLF_len-1)
            lq2_path = self.paths_lq_LLF[random_index2]
            gt2_path = self.paths_gt_LLF[random_index2]
            
            deg_type = 'LLF'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'Bokeh':
            random_index1 = random.randint(0, self.paths_Bokeh_len-1)
            lq1_path = self.paths_lq_Bokeh[random_index1]
            gt1_path = self.paths_gt_Bokeh[random_index1]
            
            random_index2 = random.randint(0, self.paths_Bokeh_len-1)
            lq2_path = self.paths_lq_Bokeh[random_index2]
            gt2_path = self.paths_gt_Bokeh[random_index2]
            
            deg_type = 'Bokeh'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'UDC_poled':
            random_index1 = random.randint(0, self.paths_UDC_poled_len-1)
            lq1_path = self.paths_lq_UDC_poled[random_index1]
            gt1_path = self.paths_gt_UDC_poled[random_index1]
            
            random_index2 = random.randint(0, self.paths_UDC_poled_len-1)
            lq2_path = self.paths_lq_UDC_poled[random_index2]
            gt2_path = self.paths_gt_UDC_poled[random_index2]
            
            deg_type = 'UDC_poled'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'UDC_toled':
            random_index1 = random.randint(0, self.paths_UDC_toled_len-1)
            lq1_path = self.paths_lq_UDC_toled[random_index1]
            gt1_path = self.paths_gt_UDC_toled[random_index1]
            
            random_index2 = random.randint(0, self.paths_UDC_toled_len-1)
            lq2_path = self.paths_lq_UDC_toled[random_index2]
            gt2_path = self.paths_gt_UDC_toled[random_index2]
            
            deg_type = 'UDC_toled'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'HistoEqu':
            random_index1 = random.randint(0, self.paths_Histo_Equ_len-1)
            lq1_path = self.paths_lq_Histo_Equ[random_index1]
            gt1_path = self.paths_gt_Histo_Equ[random_index1]
            
            random_index2 = random.randint(0, self.paths_Histo_Equ_len-1)
            lq2_path = self.paths_lq_Histo_Equ[random_index2]
            gt2_path = self.paths_gt_Histo_Equ[random_index2]
            
            deg_type = 'HistoEqu'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'ColorCorre':
            random_index1 = random.randint(0, self.paths_Color_Corre_len-1)
            lq1_path = self.paths_lq_Color_Corre[random_index1]
            gt1_path = self.paths_gt_Color_Corre[random_index1]
            
            random_index2 = random.randint(0, self.paths_Color_Corre_len-1)
            lq2_path = self.paths_lq_Color_Corre[random_index2]
            gt2_path = self.paths_gt_Color_Corre[random_index2]
            
            deg_type = 'ColorCorre'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
                
        elif dataset_choice == 'MultiTone':
            random_index1 = random.randint(0, self.paths_MultiTone_len-1)
            lq1_path = self.paths_lq_MultiTone[random_index1]
            gt1_path = self.paths_gt_MultiTone[random_index1]
            
            random_index2 = random.randint(0, self.paths_MultiTone_len-1)
            lq2_path = self.paths_lq_MultiTone[random_index2]
            gt2_path = self.paths_gt_MultiTone[random_index2]
            
            deg_type = 'MultiTone'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
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
            
        # WhiteBalance is replaced by Base. Some Problems have not been solved.
        elif dataset_choice == 'WhiteBalance':
            # multiple lq to one gt
            random_index1 = random.randint(0, self.paths_lq_WhiteBalance_len-1)
            lq1_path = self.paths_lq_WhiteBalance[random_index1]
            gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-2])
            wb_suffix = '_'.join(lq1_path.split('_')[-2:])
            gt1_path = os.path.join(self.dataset_path_gt_WhiteBalance, '{}_G_AS.png'.format(gt1_name))
            
            lq2_path_list = glob(os.path.join(self.dataset_path_lq_WhiteBalance, '*_{}'.format(wb_suffix)))
            random_index2 = random.randint(0, len(lq2_path_list)-1)
            lq2_path = lq2_path_list[random_index2]
            gt2_name = '_'.join(lq2_path.split('/')[-1].split('_')[:-2])
            gt2_path = os.path.join(self.dataset_path_gt_WhiteBalance, '{}_G_AS.png'.format(gt2_name))

            deg_type = 'WhiteBalance'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'EdgeDetec':
            random_index1 = random.randint(0, self.paths_Edge_Detect_len-1)
            lq1_path = self.paths_lq_Edge_Detect[random_index1]
            gt1_path = self.paths_gt_Edge_Detect[random_index1]
            
            random_index2 = random.randint(0, self.paths_Edge_Detect_len-1)
            lq2_path = self.paths_lq_Edge_Detect[random_index2]
            gt2_path = self.paths_gt_Edge_Detect[random_index2]
            
            deg_type = 'EdgeDetec'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
             
        elif dataset_choice == 'PencialDraw':
            random_index1 = random.randint(0, self.paths_PencilDrawing_len-1)
            lq1_path = self.paths_lq_PencilDrawing[random_index1]
            gt1_path = self.paths_gt_PencilDrawing[random_index1]
            
            random_index2 = random.randint(0, self.paths_PencilDrawing_len-1)
            lq2_path = self.paths_lq_PencilDrawing[random_index2]
            gt2_path = self.paths_gt_PencilDrawing[random_index2]
            
            deg_type = 'PencialDraw'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'DepthEstimation':
            random_index1 = random.randint(0, self.paths_DepthEstimation_len-1)
            lq1_path = self.paths_lq_DepthEstimation[random_index1]
            gt1_path = self.paths_gt_DepthEstimation[random_index1]
            
            random_index2 = random.randint(0, self.paths_DepthEstimation_len-1)
            lq2_path = self.paths_lq_DepthEstimation[random_index2]
            gt2_path = self.paths_gt_DepthEstimation[random_index2]
            
            deg_type = 'DepthEstimation'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
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
        
        elif dataset_choice == 'SaliencyObjectDetection':
            random_index1 = random.randint(0, self.paths_SaliencyObjectDetection_len-1)
            lq1_path = self.paths_lq_SaliencyObjectDetection[random_index1]
            gt1_path = self.paths_gt_SaliencyObjectDetection[random_index1]
            
            random_index2 = random.randint(0, self.paths_SaliencyObjectDetection_len-1)
            lq2_path = self.paths_lq_SaliencyObjectDetection[random_index2]
            gt2_path = self.paths_gt_SaliencyObjectDetection[random_index2]
            
            deg_type = 'SaliencyObjectDetection'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'styleClo':
            random_style_list = random.choice(self.lists_Style_Cloisonnism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Cloisonnism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Cloisonnism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleClo'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'styleDiv':
            random_style_list = random.choice(self.lists_Style_Divisionism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Divisionism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Divisionism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleDiv'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'styleFau':
            random_style_list = random.choice(self.lists_Style_Fauvism)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Fauvism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Fauvism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleFau'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'styleVermeer':
            random_style_list = random.choice(self.lists_Style_Vermeer)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Vermeer + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Vermeer + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleVermeer'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)    
                
        elif dataset_choice == 'styleJOJO':
            random_style_list = random.choice(self.lists_Style_JOJO)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_JOJO + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_JOJO + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleJOJO'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'styleRaph':
            random_style_list = random.choice(self.lists_Style_Raphael)

            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Raphael + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Raphael + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleRaph'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'styleModern':
            random_style_list = random.choice(self.lists_Style_Modernism)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Modernism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Modernism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleModern'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

        elif dataset_choice == 'styleMonet':
            random_style_list = random.choice(self.lists_Style_Monet)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Monet + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Monet + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleMonet'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'styleNeo':
            random_style_list = random.choice(self.lists_Style_NeoImpressionism)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_NeoImpressionism + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_NeoImpressionism + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleNeo'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'stylePopArt':
            random_style_list = random.choice(self.lists_Style_PopArt)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_PopArt + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_PopArt + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'stylePopArt'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'styleUkiyoe':
            random_style_list = random.choice(self.lists_Style_Ukiyoe)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_Ukiyoe + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_Ukiyoe + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleUkiyoe'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif dataset_choice == 'styleVanGogh':
            random_style_list = random.choice(self.lists_Style_VanGogh)
            
            gt1_path = random.choice(random_style_list)
            lq1_path = self.dataset_path_lq_Style_VanGogh + '/' + os.path.basename(gt1_path).split('_')[1]+'.jpg'
            gt2_path = random.choice(random_style_list)
            lq2_path = self.dataset_path_lq_Style_VanGogh + '/' + os.path.basename(gt2_path).split('_')[1]+'.jpg'
            
            deg_type = 'styleVanGogh'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()
        
        # scale = 1
        # randomly crop to designed size
        # H1, W1, C = img_lq1.shape
        # lq_size = self.gt_size // scale
        # rnd_h = random.randint(0, max(0, H1 - lq_size))
        # rnd_w = random.randint(0, max(0, W1 - lq_size))
        # img_lq1 = img_lq1[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
        # rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        # img_gt1 = img_gt1[rnd_h_gt:rnd_h_gt + self.gt_size, rnd_w_gt:rnd_w_gt + self.gt_size, :]
        
        # H2, W2, C = img_lq2.shape
        # lq_size = self.gt_size // scale
        # rnd_h = random.randint(0, max(0, H2 - lq_size))
        # rnd_w = random.randint(0, max(0, W2 - lq_size))
        # img_lq2 = img_lq2[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
        # rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        # img_gt2 = img_gt2[rnd_h_gt:rnd_h_gt + self.gt_size, rnd_w_gt:rnd_w_gt + self.gt_size, :]

        # # resize to fixed size
        # H, W, _ = img_lq1.shape
        # if H != self.gt_size or W != self.gt_size:
        #     img_lq1 = cv2.resize(img_lq1, (self.gt_size, self.gt_size),
        #                         interpolation=cv2.INTER_AREA)
            
        # H, W, _ = img_lq2.shape
        # if H != self.gt_size or W != self.gt_size:
        #     img_lq2 = cv2.resize(img_lq2, (self.gt_size, self.gt_size),
        #                         interpolation=cv2.INTER_AREA)
            
        # H, W, _ = img_gt1.shape
        # if H != self.gt_size or W != self.gt_size:
        #     img_gt1 = cv2.resize(img_gt1, (self.gt_size, self.gt_size),
        #                         interpolation=cv2.INTER_AREA)
            
        # H, W, _ = img_gt2.shape
        # if H != self.gt_size or W != self.gt_size:
        #     img_gt2 = cv2.resize(img_gt2, (self.gt_size, self.gt_size),
        #                         interpolation=cv2.INTER_AREA)
        
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
    def __init__(self, dataset_path, input_size=256, data_len=None, tasks_version=0):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.input_size = input_size
        self.data_len = data_len
        self.tasks_version = tasks_version

        random.seed(1000)
        if self.data_len is not None:
            random.shuffle(self.paths_gt)
        
        random.seed(2000)
        self.prompt_list = self.paths_gt.copy()
        random.shuffle(self.prompt_list)
        
        # dataset list   
        if self.tasks_version == 0:
            self.onthefly_degradation_list1 = ['LowLight', 'Rain', 'Ringing', 'r_l', 
                                               'Inpainting', 'Defocus_Blur', 'mosaic']
            self.onthefly_degradation_list2 = ['blur', 'noise', 'compression', 'brighten', 'darken',
                                      'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen',
                                      'saturate_weaken', 'oversharpen', 'pixelate', 'quantization']
            self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        gt1_path = self.prompt_list[idx]
        gt2_path = self.paths_gt[idx]
        
        img_gt1 = util.read_img(None, gt1_path, None, False)
        img_gt2 = util.read_img(None, gt2_path, None, False)
        
        # resize to fixed size
        H, W, _ = img_gt1.shape
        if H != self.input_size or W != self.input_size:
            img_gt1 = cv2.resize(img_gt1, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
            
        H, W, _ = img_gt2.shape
        if H != self.input_size or W != self.input_size:
            img_gt2 = cv2.resize(img_gt2, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
        
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
        
        # add on-the-fly degradation
        tasks_list = ['GenLV-Degradation-plus', 'X-Distortion', 'Operators']
        tasks_flag = np.random.choice(tasks_list, p=[1/2, 7/16, 1/16])
        
        if tasks_flag == 'GenLV-Degradation-plus':
            img_gt1 = uint2single(img_gt1)
            img_gt2 = uint2single(img_gt2)
            round_select = random.choice(['Single', 'Mix1R'])
            if round_select == 'Single':
                deg_type = random.choice(self.onthefly_degradation_list1)
                img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type)
            # 1 round Mix-degradation
            elif round_select == 'Mix1R':
                degradation_type_1 = ['LowLight', 'None', 'None', 'None', 'None']
                degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', 'None', 'None']
                degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None', 'None']
                degradation_type_4 = ['Inpainting', 'Rain', 'None', 'None', 'None']    
                degradation_type_5 = ['JPEG', 'None', 'None', 'None', 'None']
                
                deg_type1 = random.choice(degradation_type_1)
                deg_type2 = random.choice(degradation_type_2)
                deg_type3 = random.choice(degradation_type_3)
                deg_type4 = random.choice(degradation_type_4)
                deg_type5 = random.choice(degradation_type_5)
                img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type1)
                img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type2)
                img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type3)
                img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type4)
                img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type5)
                deg_type = 'Mix1R_' + deg_type1 + '_' + deg_type2 + '_' + deg_type3 + '_' + deg_type4 + '_' + deg_type5
                
        elif tasks_flag == 'X-Distortion':
            deg_type = random.choice(self.onthefly_degradation_list2)
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
                
        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
                 'input_query_img2': img_lq2, 'target_img2': img_gt2}
        return batch, deg_type

class DatasetPrompt_Customized_Val(Dataset):
    """
    Dataset for customized data, random prompt 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path_gt, dataset_path_lq, input_size=256, dataset_type='SOTS', data_len=None):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path_gt)
        self.paths_lq, self.sizes_lq = util.get_image_paths('img', dataset_path_lq)
        self.dataset_path_gt = dataset_path_gt
        self.dataset_path_lq = dataset_path_lq
        self.input_size = input_size
        sorted(self.paths_gt)
        sorted(self.paths_lq)
        self.dataset_type = dataset_type
        self.data_len = data_len
        
        random.seed(1000)
        if self.data_len is not None:
            pair_paths = list(zip(self.paths_gt, self.paths_lq))
            random.shuffle(pair_paths)
            prompt_list = pair_paths.copy()
            random.shuffle(prompt_list)
            self.paths_gt, self.paths_lq = zip(*pair_paths)
            self.prompts_gt, self.prompts_lq = zip(*prompt_list)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_lq)

    def __getitem__(self, idx):
        
        if self.dataset_type == 'SOTS':
            lq1_path = self.prompts_lq[idx]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            gt1_path = os.path.join(self.dataset_path_gt, '{}.png'.format(gt1_name))
            
            lq2_path = self.paths_lq[idx]
            gt2_name = lq2_path.split('/')[-1].split('_')[0]
            gt2_path = os.path.join(self.dataset_path_gt, '{}.png'.format(gt2_name))
            
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
            H_gt, W_gt, _ = img_gt1.shape
            H_lq, W_lq, _ = img_lq1.shape
            
            crop_size_H = np.abs(H_lq-H_gt)//2
            crop_size_W = np.abs(W_lq-W_gt)//2
            img_gt1 = img_gt1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
            img_gt2 = img_gt2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        
        elif self.dataset_type == 'InsFilterRemoval':
            lq1_path = self.prompts_lq[idx]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
            gt1_path = os.path.join(self.dataset_path_gt, '{}_Original.jpg'.format(gt1_name))
            
            gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
            lq2_name = gt2_path.split('/')[-1].split('_')[0]
            lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, filter_suffix))
            
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        elif self.dataset_type == 'InsFilterAddition':
            lq1_path = self.prompts_lq[idx]
            gt1_name = lq1_path.split('/')[-1].split('_')[0]
            filter_suffix = lq1_path.split('/')[-1].split('_')[-1]
            gt1_path = os.path.join(self.dataset_path_gt, '{}_Original.jpg'.format(gt1_name))
            
            gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
            lq2_name = gt2_path.split('/')[-1].split('_')[0]
            lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, filter_suffix))
            
            img_gt1 = util.read_img(None, lq1_path, None)
            img_lq1 = util.read_img(None, gt1_path, None) 
            img_gt2 = util.read_img(None, lq2_path, None)
            img_lq2 = util.read_img(None, gt2_path, None)
        
        elif self.dataset_type == 'ExposureError':
            lq1_path = self.prompts_lq[idx]
            gt1_name = '_'.join(lq1_path.split('/')[-1].split('_')[:-1])
            expo_suffix = lq1_path.split('_')[-1]
            gt1_path = os.path.join(self.dataset_path_gt, '{}.jpg'.format(gt1_name))
            
            gt2_path = self.paths_gt[idx] # Note: gt_idx_num < lq_idx_num 
            lq2_name = os.path.splitext(gt2_path.split('/')[-1])[0]
            lq2_path = os.path.join(self.dataset_path_lq, '{}_{}'.format(lq2_name, expo_suffix))
            
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        # validation set is not prepared.
        # elif self.dataset_type == 'WhiteBalance':
        #     lq1_path = self.prompts_lq[idx]
        #     gt1_name = lq1_path.split('/')[-1].split('_')[0]
        #     gt1_path = os.path.join(self.dataset_path_gt, '{}_G_AS.png'.format(gt1_name))
            
        #     lq2_path = self.paths_lq[idx]
        #     gt2_name = lq2_path.split('/')[-1].split('_')[0]
        #     gt2_path = os.path.join(self.dataset_path_gt, '{}_G_AS.png'.format(gt2_name))
            
        #     img_gt1 = util.read_img(None, gt1_path, None)
        #     img_lq1 = util.read_img(None, lq1_path, None) 
        #     img_gt2 = util.read_img(None, gt2_path, None)
        #     img_lq2 = util.read_img(None, lq2_path, None)
            
        else:
            gt1_path = self.prompts_gt[idx]
            lq1_path = self.prompts_lq[idx]
            
            gt2_path = self.paths_gt[idx]
            lq2_path = self.paths_lq[idx]
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
        
        deg_type = self.dataset_type
        
        # resize to fixed size
        H, W, _ = img_lq1.shape
        if H != self.input_size or W != self.input_size:
            img_lq1 = cv2.resize(img_lq1, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
            
        H, W, _ = img_lq2.shape
        if H != self.input_size or W != self.input_size:
            img_lq2 = cv2.resize(img_lq2, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
            
        H, W, _ = img_gt1.shape
        if H != self.input_size or W != self.input_size:
            img_gt1 = cv2.resize(img_gt1, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
            
        H, W, _ = img_gt2.shape
        if H != self.input_size or W != self.input_size:
            img_gt2 = cv2.resize(img_gt2, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
        
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
        
        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
                 'input_query_img2': img_lq2, 'target_img2': img_gt2,
                 'input_query_img2_path': lq2_path}
        return batch, deg_type

class DatasetPrompt_Customized_Test_DirectLoad_Triplet(Dataset):
    def __init__(self, dataset_path_root, data_len=None):
        self.dataset_path_root = dataset_path_root
        self.dataset_paths = os.listdir(self.dataset_path_root)
        sorted(self.dataset_paths)
        self.data_len = data_len
        
        random.seed(1000)
        if self.data_len is not None:
            random.shuffle(self.dataset_paths)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.dataset_paths)

    def __getitem__(self, idx):
        lq1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_input_img1.png')
        gt1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_target_img1.png')
        lq2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
        if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
            gt2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
        else:
            gt2_path = None
        
        img_gt1 = util.read_img(None, gt1_path, None)
        img_lq1 = util.read_img(None, lq1_path, None)
        
        if gt2_path:
            img_gt2 = util.read_img(None, gt2_path, None)
        
        img_lq2 = util.read_img(None, lq2_path, None)
        
        if img_gt1.ndim == 2:
            img_gt1 = np.expand_dims(img_gt1, axis=2)
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_lq1.ndim == 2:
            img_lq1 = np.expand_dims(img_lq1, axis=2)
            img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
        if gt2_path and img_gt2.ndim == 2:
            img_gt2 = np.expand_dims(img_gt2, axis=2)
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
        if img_lq2.ndim == 2:
            img_lq2 = np.expand_dims(img_lq2, axis=2)
            img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)
            
        if img_gt1.shape[2] !=3:
            img_gt1 = np.concatenate((img_gt1, img_gt1, img_gt1), axis=2)
        if img_lq1.shape[2] !=3:
            img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
        if gt2_path and img_gt2.shape[2] !=3:
            img_gt2 = np.concatenate((img_gt2, img_gt2, img_gt2), axis=2)
        if img_lq2.shape[2] !=3:
            img_lq2 = np.concatenate((img_lq2, img_lq2, img_lq2), axis=2)

        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        if gt2_path:
            img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        if gt2_path:
            target_img2 = img_gt2
        else:
            target_img2 = 'None'
        
        batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
                 'input_query_img2': img_lq2, 'target_img2': target_img2,
                 'input_query_img2_path': lq2_path}
        
        deg_type = self.dataset_paths[idx]
        return batch, deg_type
    
if __name__ == "__main__":
    dataset_train = DatasetPrompt_Train(dataset_path="data/imagenet1k/train", 
                                        input_size=256,
                                        CT_Covid_path='data/Restoration/CT_covid/train',
                                        ITS_path='data/Restoration/Dehaze/ITS', 
                                        Demoireing_path='data/Restoration/Demoireing_Processed/train',
                                        Rain13K_path='data/Restoration/Deraining/Rain13K',
                                        DustRemoval_path='data/Restoration/Dust_Removal/RB-Dust_processed/train', 
                                        Desnowing_path='data/Restoration/Desnow_CSD/Train', 
                                        FlareRemoval_path='data/Restoration/Flare7Kpp', 
                                        FaceSR_path='data/Face/ffhq', 
                                        HighlightRemoval_path='data/Restoration/highlight_removal/SHIQ_data_10825/train',
                                        LensFlare_path='data/Restoration/lens-flare', 
                                        LOL_path='data/Restoration/LOL_256/eval15', 
                                        MRI_path = 'data/MRI/train',
                                        MarineSnowRemovalSmall_path='data/Restoration/small_sized_marine_snow_removal',
                                        MarineSnowRemovalVarious_path='data/Restoration/various_sized_marine_snow_removal',
                                        ReflectionRemoval_path='data/Restoration/reflection_removal/zhang/synthetic',
                                        Satellite_path='data/Restoration/satellite/train',
                                        ShadowRemoval_ISTD_path='data/Restoration/Shadow_removal/ISTD_Dataset/train', 
                                        ShadowRemoval_ISTD_adjusted_path='data/Restoration/Shadow_removal/ISTD_adjusted', 
                                        ShadowRemoval_SRD_path='data/Restoration/Shadow_removal/SRD/Train',
                                        ThinCloudRemoval_path='data/Restoration/thin_cloud_removal/train',
                                        TextSR_train1_path='data/Restoration/TextSuperResolution-new/train/train1/png', 
                                        TextSR_train2_path='data/Restoration/TextSuperResolution-new/train/train2/png',
                                        Watermark_path='data/Restoration/watermark_removal',
                                        RealLowLightSR_path='data/Restoration/Real_Lowlight_SR',
                                        Backlit_path='data/Enhancement/backlit_image_enhancement/train/BAID_380',
                                        DSLR_blackberry_path='data/uploaded_data/DSLR/dped/blackberry/training_data',
                                        DSLR_iphone_path='data/uploaded_data/DSLR/dped/iphone/training_data',
                                        DSLR_sony_path='data/uploaded_data/DSLR/dped/sony/training_data',
                                        ExposureError_path='data/Enhancement/exposure_error/training', 
                                        InstagramFilter_path='data/Enhancement/InstagramFilterRemoval/IFFI-dataset_resort/train', 
                                        SDR_HDR_path='data/Enhancement/HDRTV1K/processed_sets', 
                                        LLF_path='data/Enhancement/LLF_256', 
                                        FiveK_path='data/Enhancement/MIT-fivek', 
                                        Bokeh_path='data/Enhancement/Rendering_Realistic_Bokeh/EBBokeh_processed/train', 
                                        Histo_Equ_path='data/Enhancement/UIEB_Dataset',
                                        UDC_poled_path='data/Enhancement/UDC/train/Poled', 
                                        UDC_toled_path='data/Enhancement/UDC/train/Toled',  
                                        WhiteBalance_path='data/Enhancement/WhiteBalance', 
                                        Color_Corre_path='data/Enhancement/UIEB_dive',
                                        VignettingRemoval_path='data/Enhancement/vignetting_removal512/train', 
                                        MultiTone_path='data/Enhancement/MIT-fivek',
                                        ISP_path='data/Enhancement/ISP/train', 
                                        Edge_Detect_path='data/Edge/BIPED/resize',
                                        DepthEstimation_path='data/ImageTranslation/depth_estimation/train',
                                        PencilDrawing_path='data/Enhancement/MIT-fivek', 
                                        Photographic_path='data/Enhancement/MIT-fivek', 
                                        RTV_path='data/Enhancement/MIT-fivek',
                                        SaliencyObjectDetection_path='data/ImageTranslation/salient_object_detection/train', 
                                        Style_Cloisonnism_path='data/Enhancement/MIT-fivek', 
                                        Style_Divisionism_path='data/Enhancement/MIT-fivek', 
                                        Style_Fauvism_path='data/Enhancement/MIT-fivek', 
                                        Style_JOJO_path='data/Enhancement/MIT-fivek', 
                                        Style_Vermeer_path='data/Enhancement/MIT-fivek', 
                                        Style_Raphael_path='data/Enhancement/MIT-fivek', 
                                        Style_Modernism_path='data/Enhancement/MIT-fivek', 
                                        Style_Monet_path='data/Enhancement/MIT-fivek',
                                        Style_NeoImpressionism_path='data/Enhancement/MIT-fivek', 
                                        Style_PopArt_path='data/Enhancement/MIT-fivek', 
                                        Style_Ukiyoe_path='data/Enhancement/MIT-fivek', 
                                        Style_VanGogh_path='data/Enhancement/MIT-fivek',
                                        data_len=None
                                        )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=6,
        drop_last=True,
    )
    
    for idx, inp_data in enumerate(data_loader_train):
        batch, degtype = inp_data
        print(degtype)
    