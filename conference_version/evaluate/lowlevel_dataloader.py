"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from evaluate.add_degradation_various import *
import dataset.util as util


class DatasetLowlevel(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, prompt_deg_type='GaussianNoise', query_deg_type='GaussianNoise',
                 padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        #self.ds = ImageFolder(os.path.join(datapath, 'val'))
        self.ds = ImageFolder(datapath)
        self.flipped_order = flipped_order
        np.random.seed(5)
        
        #print(len(self.ds))
        self.prompt_deg_type = prompt_deg_type
        print('prompt degradation type: ',  self.prompt_deg_type)
        self.query_deg_type = query_deg_type
        print('query degradation type: ',  self.query_deg_type)
        
        self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=len(self.ds)-1, replace=False)


    def __len__(self):
        return len(self.ds)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
           support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas 

    def __getitem__(self, idx):
        support_idx = np.random.choice(np.arange(0, len(self)-1))
        idx = self.indices[idx]
        query, support = self.ds[idx], self.ds[support_idx]
        
        query_img_clean = np.array(query[0])/255.0 # [0, 1]
        support_img_clean = np.array(support[0])/255.0 # [0, 1]
        
        # add noise
        if self.prompt_deg_type == 'GaussianNoise':
            support_img_deg = add_Gaussian_noise(support_img_clean.copy(), level=20)
        elif self.prompt_deg_type == 'SPNoise':
            support_img_deg = add_sp_noise(support_img_clean.copy())
        elif self.prompt_deg_type == 'iso_GaussianBlur':
            support_img_deg = iso_GaussianBlur(support_img_clean.copy(), window=15, sigma=2)
        elif self.prompt_deg_type == 'Rain':
            support_img_deg = add_rain(support_img_clean.copy(), value=200)
        elif self.prompt_deg_type == 'LowLight':
            support_img_deg = low_light(support_img_clean.copy(), lum_scale=0.3)
        elif self.prompt_deg_type == 'None':
            support_img_deg = support_img_clean
            
        if self.query_deg_type == 'GaussianNoise':
            query_img_deg = add_Gaussian_noise(query_img_clean.copy(), level=20)
        elif self.query_deg_type == 'SPNoise':
            query_img_deg = add_sp_noise(query_img_clean.copy())
        elif self.query_deg_type == 'iso_GaussianBlur':
            query_img_deg = iso_GaussianBlur(query_img_clean.copy(), window=15, sigma=2)
        elif self.query_deg_type == 'Rain':
            query_img_deg = add_rain(query_img_clean.copy(), value=200)
        elif self.query_deg_type == 'LowLight':
            query_img_deg = low_light(query_img_clean.copy(), lum_scale=0.3)
        elif self.query_deg_type == 'None':
            query_img_deg = query_img_clean
        
        query_img, query_mask = self.mask_transform(query_img_deg), self.image_transform(query_img_clean)
        support_img, support_mask = self.mask_transform(support_img_deg), self.image_transform(support_img_clean)
        
        grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                 'support_mask': support_mask, 'grid': grid}

        return batch


class DatasetLowlevel_Customized_Test_DirectLoad_Triplet(Dataset):
    def __init__(self, dataset_path_root, image_transform, mask_transform,
                padding: bool = 1, reverse_support_and_query: bool = False, flipped_order: bool = False,):
        self.dataset_path_root = dataset_path_root
        self.dataset_paths = os.listdir(self.dataset_path_root)
        sorted(self.dataset_paths)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.padding = padding
        self.reverse_support_and_query = reverse_support_and_query
        self.flipped_order = flipped_order

    def __len__(self):
        return len(self.dataset_paths)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
           support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas 

    def __getitem__(self, idx):
        LQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_input_img1.png')
        HQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_target_img1.png')
        LQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
        if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
            HQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
        else:
            HQ2_path = None
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_LQ1 = util.read_img(None, LQ1_path, None)
        
        if HQ2_path:
            img_HQ2 = util.read_img(None, HQ2_path, None)
        
        img_LQ2 = util.read_img(None, LQ2_path, None)
        
        img_HQ1 = cv2.cvtColor(img_HQ1, cv2.COLOR_BGR2RGB)
        img_LQ1 = cv2.cvtColor(img_LQ1, cv2.COLOR_BGR2RGB)
        if HQ2_path:
            img_HQ2 = cv2.cvtColor(img_HQ2, cv2.COLOR_BGR2RGB)
        img_LQ2 = cv2.cvtColor(img_LQ2, cv2.COLOR_BGR2RGB)

        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.ndim == 2:
            img_LQ1 = np.expand_dims(img_LQ1, axis=2)
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.ndim == 2:
            img_LQ2 = np.expand_dims(img_LQ2, axis=2)
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.shape[2] !=3:
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.shape[2] !=3:
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        if HQ2_path:
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()

        input_query_img1, target_img1 = self.mask_transform(img_LQ1), self.image_transform(img_HQ1)
        if HQ2_path:
            input_query_img2, target_img2 = self.mask_transform(img_LQ2), self.image_transform(img_HQ2)
        else: input_query_img2, target_img2 = self.mask_transform(img_LQ2), self.image_transform(img_LQ2)
        
        grid = self.create_grid_from_images(input_query_img1, target_img1, input_query_img2, target_img2)
        deg_type = self.dataset_paths[idx]
        
        return {'grid': grid, 'deg_type': deg_type}