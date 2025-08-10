import os
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate.add_degradation_various import *
from dataset.image_operators import *

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
    
def add_degradation_single_image(img_gt, deg_type):
    if deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq = add_Gaussian_noise(img_gt.copy(), level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq = iso_GaussianBlur(img_gt.copy(), window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq = add_JPEG_noise(img_gt.copy(), level=level)
    elif deg_type == 'Resize':
        img_lq = add_resize(img_gt.copy())
    elif deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq = add_rain(img_gt.copy(), value=value)
    elif deg_type == 'SPNoise':
        img_lq = add_sp_noise(img_gt.copy())
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq = low_light(img_gt.copy(), lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq = add_Poisson_noise(img_gt.copy(), level=2)
    elif deg_type == 'Ringing':
        img_lq = add_ringing(img_gt.copy())
    elif deg_type == 'r_l':
        img_lq = r_l(img_gt.copy())
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_lq = inpainting(img_gt.copy(), l_num=l_num, l_thick=l_thick)
    elif deg_type == 'gray':
        img_lq = cv2.cvtColor(img_gt.copy(), cv2.COLOR_BGR2GRAY)
        img_lq = np.expand_dims(img_lq, axis=2)
        img_lq = np.concatenate((img_lq, img_lq, img_lq), axis=2)

    elif deg_type == 'Laplacian':
        img_lq = img_gt.copy()
        img_gt = Laplacian_edge_detector(img_gt.copy())
    elif deg_type == 'Canny':
        img_lq = img_gt.copy()
        img_gt = Canny_edge_detector(img_gt.copy())
    elif deg_type == 'L0_smooth':
        img_lq = img_gt.copy()
        img_gt = L0_smooth(img_gt.copy())
        
    elif deg_type == 'None':
        img_lq = img_gt
    else:
        print('Error!', '-', deg_type, '-')
        exit()
    
    img_lq = np.clip(img_lq*255, 0, 255).round().astype(np.uint8)
    img_lq = img_lq.astype(np.float32)/255.0

    img_gt = np.clip(img_gt*255, 0, 255).round().astype(np.uint8)
    img_gt = img_gt.astype(np.float32)/255.0

    return img_lq, img_gt


class DatasetSinglePair_Train(Dataset):
    def __init__(self, dataset_path, input_size, ITS_path=None, LOL_path=None, Rain13K_path=None, 
                 FiveK_path=None, LLF_path=None, data_len=None, tasks_flag=None):
        
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
            
        print('dataset list: ', self.dataset_list)
        print('degradation_type list: ', self.degradation_type_list)
        
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        # dataset_choice = np.random.choice(self.dataset_list, p=[14/20, 2/20, 2/20, 2/20])
        
        if self.tasks_flag == 0:
            dataset_choice = np.random.choice(self.dataset_list, p=[7/8, 1/16, 1/16])
        elif self.tasks_flag == 1:
            dataset_choice = np.random.choice(self.dataset_list, p=[11/16, 1/16, 1/16, 1/16, 1/16, 1/16])
            
        if dataset_choice == 'Base':
            random_index = random.randint(0, self.paths_base_len-1)
            gt_path = self.paths_gt[random_index]
            img_gt = util.read_img(None, gt_path, None)
            
            # if the image size is too small
            #print(gt_size)
            H, W, _ = img_gt.shape
            #print(H, W)
            if H < self.gt_size or W < self.gt_size:
                img_gt = cv2.resize(np.copy(img_gt), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)

            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=2)
                img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)  
            if img_gt.shape[2] !=3:
                img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
            
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
                    img_lq, _ = add_degradation_single_image(np.copy(img_gt), deg_type1)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type2)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type3)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type4)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type5)
                    deg_type = 'Mix1R_' + deg_type1 + '_' + deg_type2 + '_' + deg_type3 + '_' + deg_type4 + '_' + deg_type5
                
                elif round_select == 'Single':
                    deg_type = random.choice(self.degradation_type_list)
                    img_lq, img_gt = add_degradation_single_image(img_gt, deg_type)
            
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
                    img_lq, _ = add_degradation_single_image(np.copy(img_gt), deg_type1)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type2)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type3)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type4)
                    img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type5)
                    deg_type = 'Mix1R_' + deg_type1 + '_' + deg_type2 + '_' + deg_type3 + '_' + deg_type4 + '_' + deg_type5
                
                elif round_select == 'Single':
                    deg_type = random.choice(self.degradation_type_list)
                    img_lq, img_gt = add_degradation_single_image(img_gt, deg_type)    
                
                elif round_select == 'Operator':
                    deg_type = random.choice(self.degradation_type_list2)
                    img_lq, img_gt = add_degradation_single_image(img_gt, deg_type)
                    
                    # if np.mean(img_gt).astype(np.float16) == 0:
                    #     print(deg_type, gt_path, 'zero image.')   
            
        elif dataset_choice == 'ITS':
            random_index = random.randint(0, self.paths_ITS_len-1)
            lq_path = self.paths_lq_ITS[random_index]
            gt_name = lq_path.split('/')[-1].split('_')[0]
            gt_path = os.path.join(self.dataset_path_gt_ITS, '{}.png'.format(gt_name))
            deg_type = 'ITS'
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None)
        
        elif dataset_choice == 'LOL':
            random_index = random.randint(0, self.paths_LOL_len-1)
            lq_path = self.paths_lq_LOL[random_index]
            gt_path = self.paths_gt_LOL[random_index]
            deg_type = 'LOL'
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None) 
            
        elif dataset_choice == 'Rain13K':
            random_index = random.randint(0, self.paths_Rain13K_len-1)
            lq_path = self.paths_lq_Rain13K[random_index]
            gt_path = self.paths_gt_Rain13K[random_index]
            deg_type = 'Rain13K'
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None)

            H, W, _ = img_gt.shape
            if H < self.gt_size or W < self.gt_size:
                img_gt = cv2.resize(np.copy(img_gt), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)        
            H, W, _ = img_lq.shape
            if H < self.gt_size or W < self.gt_size:
                img_lq = cv2.resize(np.copy(img_lq), (self.gt_size, self.gt_size),
                                    interpolation=cv2.INTER_LINEAR)

        elif dataset_choice == 'FiveK':
            random_index = random.randint(0, self.paths_FiveK_len-1)
            lq_path = self.paths_lq_FiveK[random_index]
            gt_path = self.paths_gt_FiveK[random_index]
            deg_type = 'FiveK'
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None) 
        
        elif dataset_choice == 'LLF':
            random_index = random.randint(0, self.paths_LLF_len-1)
            lq_path = self.paths_lq_LLF[random_index]
            gt_path = self.paths_gt_LLF[random_index]
            deg_type = 'LLF'
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None) 
            
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, self.gt_size)
        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()
        
        scale = 1
        # randomly crop to designed size
        H, W, C = img_lq.shape
        lq_size = self.gt_size // scale
        rnd_h = random.randint(0, max(0, H - lq_size))
        rnd_w = random.randint(0, max(0, W - lq_size))
        img_lq = img_lq[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
        rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        img_gt = img_gt[rnd_h_gt:rnd_h_gt + self.gt_size, rnd_w_gt:rnd_w_gt + self.gt_size, :]

        # augmentation - flip, rotate
        img_lq, img_gt = util.augment([img_lq, img_gt], hflip=True, rot=True)
        
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()
        batch = torch.stack([img_lq, img_gt], dim=0)
        return batch, deg_type
    

class DatasetSinglePair_Val(Dataset):
    def __init__(self, dataset_path, input_size, data_len=None, tasks_flag=None):
        self.paths_gt, self.sizes_gt = util.get_image_paths('img', dataset_path)
        self.input_size = input_size
        self.data_len = data_len
        self.tasks_flag = tasks_flag

        random.seed(1000)
        if self.data_len is not None:
            random.shuffle(self.paths_gt)
            
    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_gt)

    def __getitem__(self, idx):
        gt_path = self.paths_gt[idx]
        
        img_gt = util.read_img(None, gt_path, None)
        img_gt = cv2.resize(np.copy(img_gt), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
        if img_gt.shape[2] !=3:
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
        
        if self.tasks_flag == 0:
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Rain', 
                                          'SPNoise', 'PoissonNoise', 'Ringing', 'r_l', 'Inpainting']
            degradation_type_1 = ['None']
            
        elif self.tasks_flag == 1:
            self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                          'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                          'r_l', 'Inpainting', 'Laplacian', 'Canny']
            degradation_type_1 = ['LowLight', 'None']
            
        degradation_type_2 = ['GaussianBlur', 'None']
        degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', 'None']
        degradation_type_4 = ['JPEG', 'Ringing', 'r_l', 'None']
        degradation_type_5 = ['Inpainting', 'None']
        degradation_type_6 = ['Rain', 'None']
        
        round_select = np.random.choice(['Mix1R', 'Single'], p=[3/5, 2/5])
        
        if round_select == 'Mix1R':
            # 1 round Mix-degradation
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_lq, _ = add_degradation_single_image(np.copy(img_gt), deg_type1)
            img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type2)
            img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type3)
            img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type4)
            img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type5)
            img_lq, _ = add_degradation_single_image(np.copy(img_lq), deg_type6)
            deg_type = 'Mix1R_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6

        elif round_select == 'Single':
            deg_type = random.choice(self.degradation_type_list)
            img_lq, img_gt = add_degradation_single_image(img_gt, deg_type)
 
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()

        batch = {'input_query_img': img_lq, 'target_img': img_gt}
        return batch, deg_type


class DatasetSinglePair_Customized_Val(Dataset):
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
            self.paths_gt, self.paths_lq = zip(*pair_paths)

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else: return len(self.paths_lq)

    def __getitem__(self, idx):
        if self.dataset_type == 'SOTS':
            lq_path = self.paths_lq[idx]
            gt_name = lq_path.split('/')[-1].split('_')[0]
            gt_path = os.path.join(self.dataset_path_gt, '{}.png'.format(gt_name))
            
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None) 
            
            H_gt, W_gt, _ = img_gt.shape
            H_lq, W_lq, _ = img_lq.shape
            
            crop_size_H = np.abs(H_lq - H_gt) // 2
            crop_size_W = np.abs(W_lq - W_gt) // 2
            img_gt = img_gt[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        else:
            gt_path = self.paths_gt[idx]
            lq_path = self.paths_lq[idx]
            img_gt = util.read_img(None, gt_path, None)
            img_lq = util.read_img(None, lq_path, None)
        
        deg_type = self.dataset_type

        img_gt = cv2.resize(np.copy(img_gt), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_lq = cv2.resize(np.copy(img_lq), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
        if img_gt.shape[2] !=3:
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)

        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()
        
        batch = {'input_query_img': img_lq, 'target_img': img_gt, 'input_query_img_path': lq_path}
        return batch, deg_type


class DatasetSinglePair_Customized_Test_DirectLoad_Triplet(Dataset):
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
        lq_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
        if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
            gt_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
        else:
            gt_path = None

        if gt_path:
            img_gt = util.read_img(None, gt_path, None)
        img_lq = util.read_img(None, lq_path, None)

        if gt_path and img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
            img_lq = np.concatenate((img_lq, img_lq, img_lq), axis=2)

        if gt_path and img_gt.shape[2] !=3:
            img_gt = np.concatenate((img_gt, img_gt, img_gt), axis=2)
        if img_lq.shape[2] !=3:
            img_lq = np.concatenate((img_lq, img_lq, img_lq), axis=2)

        if gt_path:
            img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()
               
        input_query_img = img_lq
        if gt_path:
            target_img = img_gt
        else:
            target_img = 'None'
        
        batch = {'input_query_img': input_query_img, 'target_img': target_img, 'input_query_img_path': lq_path}
        deg_type = self.dataset_paths[idx]
        return batch, deg_type
