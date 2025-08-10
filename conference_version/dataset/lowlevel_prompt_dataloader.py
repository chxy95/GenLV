import os
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate.add_degradation_various import *
from dataset.image_operators import *

import sys
import dataset.util as util

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lq, img_gt


def add_degradation_two_images(img_gt1, img_gt2, deg_type):
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

    elif deg_type == 'Laplacian':
        img_lq1 = img_gt1.copy()
        img_gt1 = Laplacian_edge_detector(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = Laplacian_edge_detector(img_gt2.copy())
    elif deg_type == 'Canny':
        img_lq1 = img_gt1.copy()
        img_gt1 = Canny_edge_detector(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = Canny_edge_detector(img_gt2.copy())
    elif deg_type == 'Sobel':
        img_lq1 = img_gt1.copy()
        img_gt1 = Sobel_edge_detector(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = Sobel_edge_detector(img_gt2.copy())
    elif deg_type == 'L0_smooth':
        img_lq1 = img_gt1.copy()
        img_gt1 = L0_smooth(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = L0_smooth(img_gt2.copy())
    elif deg_type == 'erosion':
        img_lq1 = img_gt1.copy()
        img_gt1 = erosion(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = erosion(img_gt2.copy())
    elif deg_type == 'dilatation':
        img_lq1 = img_gt1.copy()
        img_gt1 = dilatation(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = dilatation(img_gt2.copy())
    elif deg_type == 'opening':
        img_lq1 = img_gt1.copy()
        img_gt1 = opening(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = opening(img_gt2.copy())
    elif deg_type == 'closing':
        img_lq1 = img_gt1.copy()
        img_gt1 = closing(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = closing(img_gt2.copy())
    elif deg_type == 'morphological_gradient':
        img_lq1 = img_gt1.copy()
        img_gt1 = morphological_gradient(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = morphological_gradient(img_gt2.copy())
    elif deg_type == 'top_hat':
        img_lq1 = img_gt1.copy()
        img_gt1 = top_hat(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = top_hat(img_gt2.copy())
    elif deg_type == 'black_hat':
        img_lq1 = img_gt1.copy()
        img_gt1 = black_hat(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = black_hat(img_gt2.copy())
    elif deg_type == 'embossing':
        img_lq1 = img_gt1.copy()
        img_gt1 = embossing(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = embossing(img_gt2.copy())
    elif deg_type == 'adjust_contrast':
        img_lq1 = img_gt1.copy()
        img_gt1 = adjust_contrast(img_gt1.copy())
        img_lq2 = img_gt2.copy()
        img_gt2 = adjust_contrast(img_gt2.copy())
    # elif deg_type == 'historgram_equalization':
    #     img_lq1 = img_gt1.copy()
    #     img_gt1, _ = historgram_equalization(img_gt1.copy())
    #     img_lq2 = img_gt2.copy()
    #     img_gt2, _ = historgram_equalization(img_gt2.copy())
    # elif deg_type == 'harris_corner_detection':
    #     img_lq1 = img_gt1.copy()
    #     img_gt1 = harris_corner_detection(img_gt1.copy())
    #     img_lq2 = img_gt2.copy()
    #     img_gt2 = harris_corner_detection(img_gt2.copy())
    
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
    def __init__(self, dataset_path, input_size, ITS_path=None, LOL_path=None, 
                 Rain13K_path=None, FiveK_path=None, LLF_path=None, RealOld_path=None, 
                 PaintStroke_path=None, PencilDrawing_path=None, MultiTone_path=None,
                 Photographic_path=None, style1_path=None, style2_path=None, style3_path=None,
                 style4_path=None, data_len=None, tasks_flag=None
                 ):
        
        self.data_len = data_len
        self.tasks_flag = tasks_flag
        
        np.random.seed(5)
        self.gt_size = input_size
        
        # base dataset
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.paths_base_len = len(self.paths_gt)
        
        # dehaze dataset (ITS)
        if ITS_path is not None:
            self.dataset_path_gt_ITS = os.path.join(ITS_path, 'clear')
            self.dataset_path_lq_ITS = os.path.join(ITS_path, 'hazy')
            self.paths_gt_ITS, self.sizes_gt_ITS = util.get_image_paths('img', self.dataset_path_gt_ITS)
            self.paths_lq_ITS, self.sizes_lq_ITS = util.get_image_paths('img', self.dataset_path_lq_ITS)
            self.paths_ITS_len = len(self.paths_lq_ITS)
        
        # low-light enhancement dataset (LOL)
        if LOL_path is not None:
            self.dataset_path_gt_LOL = os.path.join(LOL_path, 'high')
            self.dataset_path_lq_LOL = os.path.join(LOL_path, 'low')
            self.paths_gt_LOL, self.sizes_gt_LOL = util.get_image_paths('img', self.dataset_path_gt_LOL)
            self.paths_lq_LOL, self.sizes_lq_LOL = util.get_image_paths('img', self.dataset_path_lq_LOL)
            self.paths_LOL_len = len(self.paths_lq_LOL)
            sorted(self.paths_gt_LOL)
            sorted(self.paths_lq_LOL)
            
        # Derain dataset (Rain13K)
        if Rain13K_path is not None:
            self.dataset_path_gt_Rain13K = os.path.join(Rain13K_path, 'target')
            self.dataset_path_lq_Rain13K = os.path.join(Rain13K_path, 'input')
            self.paths_gt_Rain13K, self.sizes_gt_Rain13K = util.get_image_paths('img', self.dataset_path_gt_Rain13K)
            self.paths_lq_Rain13K, self.sizes_lq_Rain13K = util.get_image_paths('img', self.dataset_path_lq_Rain13K)
            self.paths_Rain13K_len = len(self.paths_lq_Rain13K)
            sorted(self.paths_gt_Rain13K)
            sorted(self.paths_lq_Rain13K)
            
        # Image Retouching dataset (MIT-Adobe FiveK)
        if FiveK_path is not None:
            self.dataset_path_gt_FiveK = os.path.join(FiveK_path, 'expert_C_train')
            self.dataset_path_lq_FiveK = os.path.join(FiveK_path, 'raw_input_train_png')
            self.paths_gt_FiveK, self.sizes_gt_FiveK = util.get_image_paths('img', self.dataset_path_gt_FiveK)
            self.paths_lq_FiveK, self.sizes_lq_FiveK = util.get_image_paths('img', self.dataset_path_lq_FiveK)
            self.paths_FiveK_len = len(self.paths_lq_FiveK)
            sorted(self.paths_gt_FiveK)
            sorted(self.paths_lq_FiveK)
            
        # Local Laplacian Filter dataset (MIT-Adobe FiveK - LLF)
        if LLF_path is not None:
            self.dataset_path_gt_LLF = os.path.join(LLF_path, 'expert_C_LLF_GT_train')
            self.dataset_path_lq_LLF = os.path.join(LLF_path, 'expert_C_train')
            self.paths_gt_LLF, self.sizes_gt_LLF = util.get_image_paths('img', self.dataset_path_gt_LLF)
            self.paths_lq_LLF, self.sizes_lq_LLF = util.get_image_paths('img', self.dataset_path_lq_LLF)
            self.paths_LLF_len = len(self.paths_lq_LLF)
            sorted(self.paths_gt_LLF)
            sorted(self.paths_lq_LLF)
        
        # RealOld dataset
        if RealOld_path is not None:
            self.dataset_path_gt_RealOld = os.path.join(RealOld_path, 'HQ')
            self.dataset_path_lq_RealOld = os.path.join(RealOld_path, 'LQ')
            self.paths_gt_RealOld, self.sizes_gt_RealOld = util.get_image_paths('img', self.dataset_path_gt_RealOld)
            self.paths_lq_RealOld, self.sizes_lq_RealOld = util.get_image_paths('img', self.dataset_path_lq_RealOld)
            self.paths_RealOld_len = len(self.paths_lq_RealOld)
            sorted(self.paths_gt_RealOld)
            sorted(self.paths_lq_RealOld)
        
        # PaintStroke dataset (MIT-Adobe FiveK - PaintStroke by Paint transformer: Feed forward neural painting with stroke prediction)
        if PaintStroke_path is not None:
            self.dataset_path_gt_PaintStroke = os.path.join(PaintStroke_path, 'expert_C_PaintStroke_GT_train')
            self.dataset_path_lq_PaintStroke = os.path.join(PaintStroke_path, 'expert_C_train')
            self.paths_gt_PaintStroke, self.sizes_gt_PaintStroke = util.get_image_paths('img', self.dataset_path_gt_PaintStroke)
            self.paths_lq_PaintStroke, self.sizes_lq_PaintStroke = util.get_image_paths('img', self.dataset_path_lq_PaintStroke)
            assert len(self.paths_gt_PaintStroke) == len(self.paths_lq_PaintStroke), 'Error: PaintStroke dataset length error. length of gt: {}, length of lq: {}'.format(len(self.paths_gt_PaintStroke), len(self.paths_lq_PaintStroke))
            self.paths_PaintStroke_len = len(self.paths_lq_PaintStroke)
            sorted(self.paths_gt_PaintStroke)
            sorted(self.paths_lq_PaintStroke)
        
        # PencilDrawing dataset (MIT-Adobe FiveK - PencilDrawing by Combining sketch and tone for pencil drawing production)
        if PencilDrawing_path is not None:
            self.dataset_path_gt_PencilDrawing = os.path.join(PencilDrawing_path, 'expert_C_Pencil_GT_train')
            self.dataset_path_lq_PencilDrawing = os.path.join(PencilDrawing_path, 'expert_C_train')
            self.paths_gt_PencilDrawing, self.sizes_gt_PencilDrawing = util.get_image_paths('img', self.dataset_path_gt_PencilDrawing)
            self.paths_lq_PencilDrawing, self.sizes_lq_PencilDrawing = util.get_image_paths('img', self.dataset_path_lq_PencilDrawing)
            assert len(self.paths_gt_PencilDrawing) == len(self.paths_lq_PencilDrawing), 'Error: PencilDrawing dataset length error. length of gt: {}, length of lq: {}'.format(len(self.paths_gt_PencilDrawing), len(self.paths_lq_PencilDrawing))
            self.paths_PencilDrawing_len = len(self.paths_lq_PencilDrawing)
            sorted(self.paths_gt_PencilDrawing)
            sorted(self.paths_lq_PencilDrawing)
        
        if MultiTone_path is not None:
            self.dataset_path_gt_MultiTone = os.path.join(MultiTone_path, 'expert_C_Multiscale_tone_manipulation_GT_train')
            self.dataset_path_lq_MultiTone = os.path.join(MultiTone_path, 'expert_C_train')
            self.paths_gt_MultiTone, self.sizes_gt_MultiTone = util.get_image_paths('img', self.dataset_path_gt_MultiTone)
            self.paths_lq_MultiTone, self.sizes_lq_MultiTone = util.get_image_paths('img', self.dataset_path_lq_MultiTone)
            assert len(self.paths_gt_MultiTone) == len(self.paths_lq_MultiTone), 'Error: MultiTone dataset length error. length of gt: {}, length of lq: {}'.format(len(self.paths_gt_MultiTone), len(self.paths_lq_MultiTone))
            self.paths_MultiTone_len = len(self.paths_lq_MultiTone)
            sorted(self.paths_gt_MultiTone)
            sorted(self.paths_lq_MultiTone)
        
        if Photographic_path is not None:
            self.dataset_path_gt_Photographic = os.path.join(Photographic_path, 'expert_C_Photographic_style_GT_train')
            self.dataset_path_lq_Photographic = os.path.join(Photographic_path, 'expert_C_train')
            self.paths_gt_Photographic, self.sizes_gt_Photographic = util.get_image_paths('img', self.dataset_path_gt_Photographic)
            self.paths_lq_Photographic, self.sizes_lq_Photographic = util.get_image_paths('img', self.dataset_path_lq_Photographic)
            self.paths_Photographic_len = len(self.paths_lq_Photographic)
            sorted(self.paths_gt_Photographic)
            sorted(self.paths_lq_Photographic)
            
        if style1_path is not None:
            self.dataset_path_gt_style1 = os.path.join(style1_path, 'expert_C_style1_GT_train')
            self.dataset_path_lq_style1 = os.path.join(style1_path, 'expert_C_train')
            self.paths_gt_style1, self.sizes_gt_style1 = util.get_image_paths('img', self.dataset_path_gt_style1)
            self.paths_lq_style1, self.sizes_lq_style1 = util.get_image_paths('img', self.dataset_path_lq_style1)
            self.paths_style1_len = len(self.paths_lq_style1)
            sorted(self.paths_gt_style1)
            sorted(self.paths_lq_style1)
            
        if style2_path is not None:
            self.dataset_path_gt_style2 = os.path.join(style2_path, 'expert_C_style2_GT_train')
            self.dataset_path_lq_style2 = os.path.join(style2_path, 'expert_C_train')
            self.paths_gt_style2, self.sizes_gt_style2 = util.get_image_paths('img', self.dataset_path_gt_style2)
            self.paths_lq_style2, self.sizes_lq_style2 = util.get_image_paths('img', self.dataset_path_lq_style2)
            self.paths_style2_len = len(self.paths_lq_style2)
            sorted(self.paths_gt_style2)
            sorted(self.paths_lq_style2)
            
        if style3_path is not None:
            self.dataset_path_gt_style3 = os.path.join(style3_path, 'expert_C_style3_GT_train')
            self.dataset_path_lq_style3 = os.path.join(style3_path, 'expert_C_train')
            self.paths_gt_style3, self.sizes_gt_style3 = util.get_image_paths('img', self.dataset_path_gt_style3)
            self.paths_lq_style3, self.sizes_lq_style3 = util.get_image_paths('img', self.dataset_path_lq_style3)
            self.paths_style3_len = len(self.paths_lq_style3)
            sorted(self.paths_gt_style3)
            sorted(self.paths_lq_style3)
            
        if style4_path is not None:
            self.dataset_path_gt_style4 = os.path.join(style4_path, 'expert_C_style4_GT_train')
            self.dataset_path_lq_style4 = os.path.join(style4_path, 'expert_C_train')
            self.paths_gt_style4, self.sizes_gt_style4 = util.get_image_paths('img', self.dataset_path_gt_style4)
            self.paths_lq_style4, self.sizes_lq_style4 = util.get_image_paths('img', self.dataset_path_lq_style4)
            self.paths_style4_len = len(self.paths_lq_style4)
            sorted(self.paths_gt_style4)
            sorted(self.paths_lq_style4)

        if self.tasks_flag == 0:
            self.dataset_list = ['Base', 'ITS', 'Rain13K']
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Rain', 
                                          'SPNoise', 'PoissonNoise', 'Ringing', 'r_l', 'Inpainting']
        elif self.tasks_flag == 1:
            self.dataset_list = ['Base', 'ITS', 'Rain13K', 'LOL', 'FiveK', 'LLF']
            self.degradation_type_list1 = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                          'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                          'r_l', 'Inpainting']
            self.degradation_type_list2 = ['Laplacian', 'Canny']
            self.degradation_type_list = self.degradation_type_list1 + self.degradation_type_list2
        else:
            self.dataset_list = ['Base', 'ITS', 'Rain13K', 'LOL', 'FiveK', 'LLF', 
                                'PaintStroke', 'PencilDrawing', 'MultiTone', 'Photographic', 
                                'style1', 'style2', 'style3', 'style4']
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                    'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting', 'Laplacian', 'Canny',
                                    'Sobel', 'erosion', 'dilatation', 'opening',
                                    'closing', 'morphological_gradient', 'adjust_contrast',]
        
        print('dataset list: ', self.dataset_list)
        print('degradation_type list: ', self.degradation_type_list)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        # dataset_choice = np.random.choice(self.dataset_list, p=[10/20, 2/20, 2/20, 2/20, 2/20, 2/20])
        # dataset_choice = np.random.choice(self.dataset_list, p=[19/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, 1/32, ])
        # dataset_choice = np.random.choice(self.dataset_list, p=[3/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, ])
        
        if self.tasks_flag == 0:
            dataset_choice = np.random.choice(self.dataset_list, p=[7/8, 1/16, 1/16])
        elif self.tasks_flag == 1:
            dataset_choice = np.random.choice(self.dataset_list, p=[11/16, 1/16, 1/16, 1/16, 1/16, 1/16])
        else: # need further design
            dataset_choice = np.random.choice(self.dataset_list, p=[3/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, ])
            
        if dataset_choice == 'Base':
            random_index1 = random.randint(0, self.paths_base_len-1)
            gt1_path = self.paths_gt[random_index1]
            random_index2 = random.randint(0, self.paths_base_len-1)
            gt2_path = self.paths_gt[random_index2]
            
            img_gt1 = util.read_img(None, gt1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)

            # if the image size is too small
            H, W, _ = img_gt1.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt1 = cv2.resize(np.copy(img_gt1), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)
                
            H, W, _ = img_gt2.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt2 = cv2.resize(np.copy(img_gt2), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)
                
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
            
            # add degradation to gt
            if self.tasks_flag == 0:
                degradation_type_1 = ['None']
                degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', 'None']
                degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None']
                degradation_type_4 = ['JPEG', 'None']
                degradation_type_5 = ['Inpainting', 'Rain', 'None']
            
                round_select = np.random.choice(['Mix1R', 'Single'], p=[3/5, 2/5]) # original version: [4/5, 1/5]
                
                if round_select == 'Mix1R':
                    # 1 round Mix-degradation
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

                elif round_select == 'Single':
                    deg_type = random.choice(self.degradation_type_list)
                    img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(img_gt1, img_gt2, deg_type)
                            
            elif self.tasks_flag == 1:
                degradation_type_1 = ['LowLight', 'None', 'None', 'None', 'None']
                degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', 'None', 'None']
                degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None', 'None']
                degradation_type_4 = ['JPEG', 'None', 'None', 'None', 'None']
                degradation_type_5 = ['Inpainting', 'Rain', 'None', 'None', 'None']
            
                round_select = np.random.choice(['Mix1R', 'Single', 'Operator'], p=[1/2, 1/4, 1/4])
                
                if round_select == 'Mix1R':
                    # 1 round Mix-degradation
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

                elif round_select == 'Single':
                    deg_type = random.choice(self.degradation_type_list1)
                    img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(img_gt1, img_gt2, deg_type)

                elif round_select == 'Operator':
                    deg_type = random.choice(self.degradation_type_list2)
                    img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(img_gt1, img_gt2, deg_type)
                    
                    if np.mean(img_gt1).astype(np.float16) == 0 or np.mean(img_gt2).astype(np.float16) == 0:
                        # print(deg_type, gt1_path, gt2_path)
                        if np.mean(img_gt1).astype(np.float16) == 0 and np.mean(img_gt2).astype(np.float16) == 0:
                            print(deg_type, gt1_path, gt2_path, 'two zero images.')
                        if np.mean(img_gt1).astype(np.float16) == 0:
                            print(deg_type, gt1_path, 'zero image.')
                            img_gt1 = img_gt2.copy()
                            img_lq1 = img_lq2.copy()
                        if np.mean(img_gt2).astype(np.float16) == 0:
                            print(deg_type, gt2_path, 'zero image.')
                            img_gt2 = img_gt1.copy()
                            img_lq2 = img_lq1.copy()
            
            else:
                degradation_type_1 = ['None']
                degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', 'None']
                degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None']
                degradation_type_4 = ['JPEG', 'None']
                degradation_type_5 = ['Inpainting', 'Rain', 'None']
            
                round_select = np.random.choice(['Mix1R', 'Single'], p=[4/5, 1/5])
                
                if round_select == 'Mix1R':
                    # 1 round Mix-degradation
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

                elif round_select == 'Single':
                    deg_type = random.choice(self.degradation_type_list)
                    img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(img_gt1, img_gt2, deg_type)
                    
                    if np.mean(img_gt1).astype(np.float16) == 0 or np.mean(img_gt2).astype(np.float16) == 0:
                        # print(deg_type, gt1_path, gt2_path)
                        if np.mean(img_gt1).astype(np.float16) == 0 and np.mean(img_gt2).astype(np.float16) == 0:
                            print(deg_type, gt1_path, gt2_path, 'two zero images.')
                        if np.mean(img_gt1).astype(np.float16) == 0:
                            print(deg_type, gt1_path, 'zero image.')
                            img_gt1 = img_gt2.copy()
                            img_lq1 = img_lq2.copy()
                        if np.mean(img_gt2).astype(np.float16) == 0:
                            print(deg_type, gt2_path, 'zero image.')
                            img_gt2 = img_gt1.copy()
                            img_lq2 = img_lq1.copy()
                            
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
        
        elif dataset_choice == 'PaintStroke':
            random_index1 = random.randint(0, self.paths_PaintStroke_len-1)
            lq1_path = self.paths_lq_PaintStroke[random_index1]
            gt1_path = self.paths_gt_PaintStroke[random_index1]
            
            random_index2 = random.randint(0, self.paths_PaintStroke_len-1)
            lq2_path = self.paths_lq_PaintStroke[random_index2]
            gt2_path = self.paths_gt_PaintStroke[random_index2]
            
            deg_type = 'PaintStroke'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
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
            
        elif dataset_choice == 'style1':
            random_index1 = random.randint(0, self.paths_style1_len-1)
            lq1_path = self.paths_lq_style1[random_index1]
            gt1_path = self.paths_gt_style1[random_index1]
            
            random_index2 = random.randint(0, self.paths_style1_len-1)
            lq2_path = self.paths_lq_style1[random_index2]
            gt2_path = self.paths_gt_style1[random_index2]
            
            deg_type = 'style1'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'style2':
            random_index1 = random.randint(0, self.paths_style2_len-1)
            lq1_path = self.paths_lq_style2[random_index1]
            gt1_path = self.paths_gt_style2[random_index1]
            
            random_index2 = random.randint(0, self.paths_style2_len-1)
            lq2_path = self.paths_lq_style2[random_index2]
            gt2_path = self.paths_gt_style2[random_index2]
            
            deg_type = 'style2'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'style3':
            random_index1 = random.randint(0, self.paths_style3_len-1)
            lq1_path = self.paths_lq_style3[random_index1]
            gt1_path = self.paths_gt_style3[random_index1]
            
            random_index2 = random.randint(0, self.paths_style3_len-1)
            lq2_path = self.paths_lq_style3[random_index2]
            gt2_path = self.paths_gt_style3[random_index2]
            
            deg_type = 'style3'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None)
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)
            
        elif dataset_choice == 'style4':
            random_index1 = random.randint(0, self.paths_style4_len-1)
            lq1_path = self.paths_lq_style4[random_index1]
            gt1_path = self.paths_gt_style4[random_index1]
            
            random_index2 = random.randint(0, self.paths_style4_len-1)
            lq2_path = self.paths_lq_style4[random_index2]
            gt2_path = self.paths_gt_style4[random_index2]
            
            deg_type = 'style4'
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
            
            H, W, _ = img_gt1.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt1 = cv2.resize(np.copy(img_gt1), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)
            H, W, _ = img_gt2.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt2 = cv2.resize(np.copy(img_gt2), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)              
            H, W, _ = img_lq1.shape
            #print(H, W)
            if H < self.gt_size or W < self.gt_size:
                img_lq1 = cv2.resize(np.copy(img_lq1), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)
            H, W, _ = img_lq2.shape
            if H < self.gt_size or W < self.gt_size:
                img_lq2 = cv2.resize(np.copy(img_lq2), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)
        
        elif dataset_choice == 'RealOld':
            random_index1 = random.randint(0, self.paths_RealOld_len-1)
            lq1_path = self.paths_lq_RealOld[random_index1]
            gt1_path = self.paths_gt_RealOld[random_index1]
            
            random_index2 = random.randint(0, self.paths_RealOld_len-1)
            lq2_path = self.paths_lq_RealOld[random_index2]
            gt2_path = self.paths_gt_RealOld[random_index2]
            
            deg_type = 'RealOld'
            img_gt1 = util.read_img(None, gt1_path, None)
            img_lq1 = util.read_img(None, lq1_path, None) 
            img_gt2 = util.read_img(None, gt2_path, None)
            img_lq2 = util.read_img(None, lq2_path, None)

            H, W, _ = img_gt1.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt1 = cv2.resize(np.copy(img_gt1), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)   
            H, W, _ = img_gt2.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt2 = cv2.resize(np.copy(img_gt2), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)    
            H, W, _ = img_lq1.shape
            if H < self.gt_size or W < self.gt_size:
                img_lq1 = cv2.resize(np.copy(img_lq1), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)    
            H, W, _ = img_lq2.shape
            if H < self.gt_size or W < self.gt_size:
                img_lq2 = cv2.resize(np.copy(img_lq2), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)        
        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()
        
        scale = 1
        # randomly crop to designed size
        H1, W1, C = img_lq1.shape
        lq_size = self.gt_size // scale
        rnd_h = random.randint(0, max(0, H1 - lq_size))
        rnd_w = random.randint(0, max(0, W1 - lq_size))
        img_lq1 = img_lq1[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
        rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        img_gt1 = img_gt1[rnd_h_gt:rnd_h_gt + self.gt_size, rnd_w_gt:rnd_w_gt + self.gt_size, :]
        
        H2, W2, C = img_lq2.shape
        lq_size = self.gt_size // scale
        rnd_h = random.randint(0, max(0, H2 - lq_size))
        rnd_w = random.randint(0, max(0, W2 - lq_size))
        img_lq2 = img_lq2[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
        rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        img_gt2 = img_gt2[rnd_h_gt:rnd_h_gt + self.gt_size, rnd_w_gt:rnd_w_gt + self.gt_size, :]

        # augmentation - flip, rotate
        img_lq1, img_lq2, img_gt1, img_gt2 = util.augment([img_lq1, img_lq2, img_gt1, img_gt2], hflip=True, rot=True)
        
        if not np.all(np.isfinite(np.concatenate((img_lq1, img_lq2, img_gt1, img_gt2), axis=2))):
            print("Data exists unfinite value.")
            sys.exit(1)

        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = torch.stack([img_lq1, img_gt1, img_lq2, img_gt2], dim=0)
        return batch, deg_type
    

class DatasetPrompt_Val(Dataset):
    def __init__(self, dataset_path, input_size=320, data_len=None, tasks_flag=None):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.input_size = input_size
        self.data_len = data_len
        self.tasks_flag = tasks_flag

        random.seed(1000)
        if self.data_len is not None:
            random.shuffle(self.paths_gt)
        
        random.seed(2000)
        self.prompt_list = self.paths_gt.copy()
        random.shuffle(self.prompt_list)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        gt1_path = self.prompt_list[idx]
        gt2_path = self.paths_gt[idx]
        
        img_gt1 = util.read_img(None, gt1_path, None)
        img_gt2 = util.read_img(None, gt2_path, None)
        
        img_gt1 = cv2.resize(np.copy(img_gt1), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        img_gt2 = cv2.resize(np.copy(img_gt2), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        
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
        
        
        if self.tasks_flag == 0:
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Rain', 
                                          'SPNoise', 'PoissonNoise', 'Ringing', 'r_l', 'Inpainting']
            degradation_type_1 = ['None']
            
        elif self.tasks_flag == 1:
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                          'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                          'r_l', 'Inpainting', 'Laplacian', 'Canny']
            degradation_type_1 = ['LowLight', 'None']
            
        else:
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                    'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting', 'Laplacian', 'Canny',
                                    'Sobel', 'erosion', 'dilatation', 'opening',
                                    'closing', 'morphological_gradient', 'adjust_contrast']
            degradation_type_1 = ['LowLight', 'None']
            
        degradation_type_2 = ['GaussianBlur', 'None']
        degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None']
        degradation_type_4 = ['JPEG', 'Ringing', 'r_l', 'None']
        degradation_type_5 = ['Inpainting', 'None']
        degradation_type_6 = ['Rain', 'None']
        
        round_select = np.random.choice(['Mix1R', 'Single'], p=[3/5, 2/5]) # original version: [4/5, 1/5]
        
        if round_select == 'Mix1R':
            # 1 round Mix-degradation
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_gt1), np.copy(img_gt2), deg_type1)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type2)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type3)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type4)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type5)
            img_lq1, img_lq2, _, _ = add_degradation_two_images(np.copy(img_lq1), np.copy(img_lq2), deg_type6)
            deg_type = 'Mix1R_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6

        elif round_select == 'Single':
            deg_type = random.choice(self.degradation_type_list)
            img_lq1, img_lq2, img_gt1, img_gt2 = add_degradation_two_images(img_gt1, img_gt2, deg_type)
            
        # elif round_select == 'Mix2R':
        #     # 1 round Mix-degradation (currently does not take into account, 1 round is enough)
        #     deg_type1 = random.choice(degradation_type_1)
        #     deg_type2 = random.choice(degradation_type_2)
        #     deg_type3 = random.choice(degradation_type_3)
        #     deg_type4 = random.choice(degradation_type_4)
        #     deg_type5 = random.choice(degradation_type_5)
        #     deg_type6 = random.choice(degradation_type_6)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_gt1, img_gt2, deg_type1)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type2)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type3)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type4)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type5)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type6)
            
        #     deg_type7 = random.choice(degradation_type_1)
        #     deg_type8 = random.choice(degradation_type_2)
        #     deg_type9 = random.choice(degradation_type_3)
        #     deg_type10 = random.choice(degradation_type_4)
        #     deg_type11 = random.choice(degradation_type_5)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type7)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type8)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type9)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type10)
        #     img_lq1, img_lq2, _, _ = add_degradation_two_images(img_lq1, img_lq2, deg_type11)
            
        #     deg_type = 'R2_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6+'_'+ \
        #                 deg_type7+'_'+deg_type8+'_'+deg_type9+'_'+deg_type10+'_'+deg_type11

        img_gt1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt1, (2, 0, 1)))).float()
        img_gt2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt2, (2, 0, 1)))).float()
        img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()
        img_lq2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq2, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': img_lq1, 'target_img1': img_gt1,
                 'input_query_img2': img_lq2, 'target_img2': img_gt2}
        return batch, deg_type


class DatasetPrompt_Customized_Val(Dataset):
    def __init__(self, dataset_path_gt, dataset_path_lq, dataset_type='SOTS', data_len=None):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path_gt)
        self.paths_lq, self.sizes_lq = util.get_image_paths('img', dataset_path_lq)
        self.dataset_path_gt = dataset_path_gt
        self.dataset_path_lq = dataset_path_lq
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
        img_gt1 = cv2.resize(np.copy(img_gt1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_lq1 = cv2.resize(np.copy(img_lq1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_gt2 = cv2.resize(np.copy(img_gt2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_lq2 = cv2.resize(np.copy(img_lq2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        
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