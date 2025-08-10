import random
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import dataset.util as util


class DatasetPrompt_Synthetic_Test(Dataset):
    """
    Dataset for Synthetic Test Dataset, fixed prompt 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path_inp, dataset_path_out, prompt_path='', task_type='SRx2'):
        self.prompt_path = prompt_path
        self.task_type = task_type
        
        self.paths_out, self.sizes_out = util.get_image_paths('img', dataset_path_out)
        self.paths_inp, self.sizes_inp = util.get_image_paths('img', dataset_path_inp)

        # read prompt images
        self.onthefly_task_list1 = ['Rain', 'Ringing', 'r_l', 'Inpainting', 'mosaic', 'SRx2', 'SRx4']
        self.onthefly_task_list2 = ['Laplacian', 'Canny']
        self.onthefly_task_list3 = ['blur_gaussian', 'blur_motion', 'blur_glass', 'blur_lens', 'blur_zoom', 'blur_jitter', 
                                    'noise_speckle', 'noise_spatially_correlated', 'noise_poisson', 'noise_impulse', 
                                    'compression_jpeg', 'compression_jpeg_2000', 'oversharpen', 'pixelate', 
                                    'quantization_otsu', 'quantization_median', 'quantization_hist', 'spatter', 
                                    'blur_gaussian_lensmask']
        self.onthefly_task_list4 = ['noise_gaussian', 'brightness_brighten', 'brightness_darken', 
                                    'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen', 'saturate_weaken']
        
        # read prompt images.
        self.prompt_out_img = util.read_img(None, osp.join(prompt_path, 'target_256x256.png'), None)
        if task_type in self.onthefly_task_list1:
            self.prompt_inp_img = util.read_img(None, osp.join(prompt_path, task_type+'.png'), None)
        elif task_type in self.onthefly_task_list2:
            self.prompt_inp_img = self.prompt_out_img.copy()
            self.prompt_out_img = util.read_img(None, osp.join(prompt_path, task_type+'.png'), None)
        elif task_type in self.onthefly_task_list3:
            self.prompt_inp_imgs = [util.read_img(None, osp.join(prompt_path, task_type+'_{}.png'.format(i)), None) for i in [1,2,3,4,5]]
        elif task_type in self.onthefly_task_list4:
            pass
        elif task_type in ['OOD_gaussian_noise']:
            self.prompt_inp_imgs = [util.read_img(None, osp.join(prompt_path, task_type+'_{}.png'.format(i)), None) for i in [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.35, 0.50]]
        elif task_type in ['OOD_blur_gaussian']:
            self.prompt_inp_imgs = [util.read_img(None, osp.join(prompt_path, task_type+'_{}.png'.format(i)), None) for i in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]]
            
    def __len__(self):
        return len(self.paths_inp)

    def __getitem__(self, idx):
        if self.task_type not in self.onthefly_task_list2:
            out_path = self.paths_out[idx]
            inp_path = self.paths_inp[idx]
        else:
            out_path = self.paths_inp[idx]
            inp_path = self.paths_out[idx]
        
        img_out = util.read_img(None, out_path, None)
        img_inp = util.read_img(None, inp_path, None) 
        prompt_out = self.prompt_out_img.copy()
        
        if self.task_type in self.onthefly_task_list1 or self.task_type in self.onthefly_task_list2:
            prompt_inp = self.prompt_inp_img.copy()
            deg_type_with_severity = self.task_type
            
        elif self.task_type in self.onthefly_task_list3:
            severity = int(inp_path.split('_')[-1][0])
            prompt_inp = self.prompt_inp_imgs[severity-1].copy()
            deg_type_with_severity = '_'.join([self.task_type, str(severity)])
            
        elif self.task_type in self.onthefly_task_list4:
            prompt_inp_path = '_'.join(osp.basename(inp_path).split('_')[1:])
            prompt_inp = util.read_img(None, osp.join(self.prompt_path, prompt_inp_path), None)
            deg_type_with_severity = prompt_inp_path[:-4]
            
        elif self.task_type == 'OOD_gaussian_noise':
            sigma = float(inp_path.split('_')[-1][:-4])
            prompt_inp = self.prompt_inp_imgs[[0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.35, 0.50].index(sigma)].copy()
            deg_type_with_severity = '_'.join([self.task_type, str(sigma)])
        
        elif self.task_type == 'OOD_blur_gaussian':
            sigma = float(inp_path.split('_')[-1][:-4])
            prompt_inp = self.prompt_inp_imgs[[0.5, 1.5, 2.5, 3.5, 4.5, 5.5].index(sigma)].copy()
            deg_type_with_severity = '_'.join([self.task_type, str(sigma)])
                
        prompt_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_inp, (2, 0, 1)))).float()
        prompt_out = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_out, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()
        img_out = torch.from_numpy(np.ascontiguousarray(np.transpose(img_out, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': prompt_inp, 'target_img1': prompt_out,
                 'input_query_img2': img_inp, 'target_img2': img_out,
                 'input_query_img2_path': inp_path}
        return batch, deg_type_with_severity
    

class DatasetPrompt_Customize_Prompt(Dataset):
    """
    Dataset for Customize Prompt Test

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, prompt_inp_path, prompt_out_path, dataset_inp_path='', dataset_out_path=None, task_info=''):
        self.paths_inp, _ = util.get_image_paths('img', dataset_inp_path)
        if dataset_out_path is not None:
            self.paths_out, _ = util.get_image_paths('img', dataset_out_path)
        else: self.paths_out = None
        
        self.task_info = task_info
        
        # read prompt images.
        prompt_inp_img = util.read_img(None, prompt_inp_path, None)
        H, W, _ = prompt_inp_img.shape
        if H < 256 or W < 256:
            self.prompt_inp_img = cv2.resize(prompt_inp_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            self.prompt_inp_img = cv2.resize(prompt_inp_img, (256, 256), interpolation=cv2.INTER_AREA)

        prompt_out_img = util.read_img(None, prompt_out_path, None)
        H, W, _ = prompt_out_img.shape
        if H < 256 or W < 256:
            self.prompt_out_img = cv2.resize(prompt_out_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            self.prompt_out_img = cv2.resize(prompt_out_img, (256, 256), interpolation=cv2.INTER_AREA)
            
    def __len__(self):
        return len(self.paths_inp)

    def __getitem__(self, idx):
        inp_path = self.paths_inp[idx]
        img_inp = util.read_img(None, inp_path, None)
        H, W, _ = img_inp.shape
        if H < 256 or W < 256:
            img_inp = cv2.resize(img_inp, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            img_inp = cv2.resize(img_inp, (256, 256), interpolation=cv2.INTER_AREA)
        
        if self.paths_out is not None:
            out_path = self.paths_out[idx]
            img_out = util.read_img(None, out_path, None)
            H, W, _ = img_out.shape
            if H < 256 or W < 256:
                img_out = cv2.resize(img_out, (256, 256), interpolation=cv2.INTER_CUBIC)
            else: 
                img_out = cv2.resize(img_out, (256, 256), interpolation=cv2.INTER_AREA)
        
        prompt_inp = self.prompt_inp_img.copy()
        prompt_out = self.prompt_out_img.copy()

        prompt_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_inp, (2, 0, 1)))).float()
        prompt_out = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_out, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()
        if self.paths_out is not None:
            img_out = torch.from_numpy(np.ascontiguousarray(np.transpose(img_out, (2, 0, 1)))).float()
        else: img_out = 'None'
        
        batch = {'input_query_img1': prompt_inp, 'target_img1': prompt_out,
                 'input_query_img2': img_inp, 'target_img2': img_out,
                 'input_query_img2_path': inp_path}
        return batch, self.task_info
    
    
class DatasetPrompt_Individual_Test(Dataset):
    """
    Dataset for Individual Test Dataset, fixed prompt 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, task_type=''):        
        task_list1 = ['Derain', 'RainDrop', 'MarineSnowRemoval', 'ReflectionRemoval', 'CloudRemoval', 'WatermarkRemoval', 
                  'UDCPoled', 'UDCToled', 'Demoireing', 'DustRemoval', 'Desnow', 'FlareRemoval', 'HighlightRemoval'] # 13
        task_list2 = ['LowLight', 'BacklitEnhance', 'ISP', 'RenderBokeh', 'VignettingRemoval', 'PhotoRetouch'] # 6
        task_list3 = ['LocalLapFilter', 'MultiScaleTM'] # 2
        task_list4 = ['PencilDrawing', 'Photographic', 'RTV'] # 3
        task_list5 = ['Face-SR', 'Infrared-SR', 'CT-SR', 'CT-Denoise', 'MRI-SR', 'MRI-Denoise', 'Satellite-SR', 'Satellite-Denoise'] # 8
        task_list6 = ['Weather-SR', 'FluidFlow-SR', 'Cosmology-SR'] # 3
        task_list7 = ['DepthEstimate', 'PercepEdgeDetect', 'SaliencyObject', 'HoughLine', 'Normal', 'HEDBoundary'] # 6
        task_list8 = ['Cloisonnism', 'Divisionism', 'Fauvism', 'Vermeer', 'JOJO', 'Raphael', 'Modernism', 'Monet',
                    'NeoImpressionism', 'PopArt', 'Ukiyoe', 'VanGogh', 'Tuner', 'Regionalism', 'Impressionism'] # 15

        PhotoRetouch_path = '/cpfs01/user/puyuandong/glv/genlv/data/GenLV_Processed_Data/Enhancement/PhotoRetouch'
        if task_type in task_list1 + task_list2 + task_list7:
            self.paths_inp, self.size_inp = util.get_image_paths('img', osp.join(dataset_path, 'Test_256/input'))
            self.paths_out, self.size_out = util.get_image_paths('img', osp.join(dataset_path, 'Test_256/target'))
        elif task_type in task_list3 + task_list4 + task_list8:
            self.paths_inp, self.size_inp = util.get_image_paths('img', osp.join(PhotoRetouch_path, 'Test_256/target'))
            self.paths_out, self.size_out = util.get_image_paths('img', osp.join(dataset_path, 'Test_256_target'))
        elif task_type in task_list5 + task_list6:
            name, task = task_type.split('-')
            self.paths_out, self.size_out = util.get_image_paths('img', osp.join(dataset_path, name, 'Test_target_256'))
            if task == 'SR':
                self.paths_inp, self.size_inp = util.get_image_paths('img', osp.join(dataset_path, name, 'Test_X4Down_256'))
            elif task == 'Denoise':
                self.paths_inp, self.size_inp = util.get_image_paths('img', osp.join(dataset_path, name, 'Test_GN15_256'))
        
        if self.size_inp != self.size_out:
            assert('Wrong sizes of input / target folder.')
        
        random.seed(0)
        if task_type not in task_list8:
            prompt_inp_path = random.choice(self.paths_inp)
            prompt_out_path = self.paths_out[self.paths_inp.index(prompt_inp_path)]
            self.prompt_inp_img = util.read_img(None, prompt_inp_path, None)
            self.prompt_out_img = util.read_img(None, prompt_out_path, None)
            self.task_info = task_type + '=' + osp.basename(prompt_inp_path)[:-4]
            self.style_flag = False
        else:
            style1 = [p for p in self.paths_out if osp.basename(p)[:2]=='1_']
            style2 = [p for p in self.paths_out if osp.basename(p)[:2]=='2_']
            style3 = [p for p in self.paths_out if osp.basename(p)[:2]=='3_']
            style4 = [p for p in self.paths_out if osp.basename(p)[:2]=='4_']
            style5 = [p for p in self.paths_out if osp.basename(p)[:2]=='5_']
            
            prompt_out_path1 = random.choice(style1)
            prompt_inp_path1 = self.paths_inp[self.paths_out.index(prompt_out_path1)]
            self.prompt_inp_img1 = util.read_img(None, prompt_inp_path1, None)
            self.prompt_out_img1 = util.read_img(None, prompt_out_path1, None)
            
            prompt_out_path2 = random.choice(style2)
            prompt_inp_path2 = self.paths_inp[self.paths_out.index(prompt_out_path2)]
            self.prompt_inp_img2 = util.read_img(None, prompt_inp_path2, None)
            self.prompt_out_img2 = util.read_img(None, prompt_out_path2, None)

            prompt_out_path3 = random.choice(style3)
            prompt_inp_path3 = self.paths_inp[self.paths_out.index(prompt_out_path3)]
            self.prompt_inp_img3 = util.read_img(None, prompt_inp_path3, None)
            self.prompt_out_img3 = util.read_img(None, prompt_out_path3, None)
            
            prompt_out_path4 = random.choice(style4)
            prompt_inp_path4 = self.paths_inp[self.paths_out.index(prompt_out_path4)]
            self.prompt_inp_img4 = util.read_img(None, prompt_inp_path4, None)
            self.prompt_out_img4 = util.read_img(None, prompt_out_path4, None)
            
            prompt_out_path5 = random.choice(style5)
            prompt_inp_path5 = self.paths_inp[self.paths_out.index(prompt_out_path5)]
            self.prompt_inp_img5 = util.read_img(None, prompt_inp_path5, None)
            self.prompt_out_img5 = util.read_img(None, prompt_out_path5, None)
            self.style_flag = True
            self.task_info = [task_type,
                              osp.basename(prompt_inp_path1)[:-4],
                              osp.basename(prompt_inp_path2)[:-4],
                              osp.basename(prompt_inp_path3)[:-4],
                              osp.basename(prompt_inp_path4)[:-4],
                              osp.basename(prompt_inp_path5)[:-4],
                             ]
            
    def __len__(self):
        return len(self.paths_inp)

    def __getitem__(self, idx):
        inp_path = self.paths_inp[idx]
        img_inp = util.read_img(None, inp_path, None)
        
        out_path = self.paths_out[idx]
        img_out = util.read_img(None, out_path, None)
        
        if self.style_flag: 
            if osp.basename(out_path)[:2]=='1_':
                prompt_inp = self.prompt_inp_img1.copy()
                prompt_out = self.prompt_out_img1.copy()
            elif osp.basename(out_path)[:2]=='2_':
                prompt_inp = self.prompt_inp_img2.copy()
                prompt_out = self.prompt_out_img2.copy()
            elif osp.basename(out_path)[:2]=='3_':
                prompt_inp = self.prompt_inp_img3.copy()
                prompt_out = self.prompt_out_img3.copy()
            elif osp.basename(out_path)[:2]=='4_':
                prompt_inp = self.prompt_inp_img4.copy()
                prompt_out = self.prompt_out_img4.copy()
            elif osp.basename(out_path)[:2]=='5_':
                prompt_inp = self.prompt_inp_img5.copy()
                prompt_out = self.prompt_out_img5.copy()
        else:
            prompt_inp = self.prompt_inp_img.copy()
            prompt_out = self.prompt_out_img.copy()

        prompt_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_inp, (2, 0, 1)))).float()
        prompt_out = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_out, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()
        img_out = torch.from_numpy(np.ascontiguousarray(np.transpose(img_out, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': prompt_inp, 'target_img1': prompt_out,
                 'input_query_img2': img_inp, 'target_img2': img_out,
                 'input_query_img2_path': inp_path}
        return batch, self.task_info
    
    
class DatasetPrompt_General(Dataset):  # by zkw, adapted from DatasetPrompt_Customize_Prompt
    """
    Dataset for General Prompt Test that supports customized input->target name mapping

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, prompt_inp_path, prompt_out_path, dataset_inp_path='', dataset_out_path=None, task_info='',
                 inp_tgt_map=lambda x: x):
        self.paths_inp, _ = util.get_image_paths('img', dataset_inp_path)
        # if dataset_out_path is not None:
        #     self.paths_out, _ = util.get_image_paths('img', dataset_out_path)
        # else: self.paths_out = None
        self.tgt_dir = Path(dataset_out_path)
        
        self.task_info = task_info
        self.inp_tgt_map = inp_tgt_map
        
        # read prompt images.
        prompt_inp_img = util.read_img(None, prompt_inp_path, None)
        H, W, _ = prompt_inp_img.shape
        if H < 256 or W < 256:
            self.prompt_inp_img = cv2.resize(prompt_inp_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            self.prompt_inp_img = cv2.resize(prompt_inp_img, (256, 256), interpolation=cv2.INTER_AREA)

        prompt_out_img = util.read_img(None, prompt_out_path, None)
        H, W, _ = prompt_out_img.shape
        if H < 256 or W < 256:
            self.prompt_out_img = cv2.resize(prompt_out_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            self.prompt_out_img = cv2.resize(prompt_out_img, (256, 256), interpolation=cv2.INTER_AREA)
            
    def __len__(self):
        return len(self.paths_inp)

    def __getitem__(self, idx):
        inp_path = self.paths_inp[idx]
        img_inp = util.read_img(None, inp_path, None)
        H, W, _ = img_inp.shape
        if H < 256 or W < 256:
            img_inp = cv2.resize(img_inp, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            img_inp = cv2.resize(img_inp, (256, 256), interpolation=cv2.INTER_AREA)
        
        # if self.paths_out is not None:
        
        # out_path = self.paths_out[idx]
        img_name = osp.basename(inp_path)
        out_path = self.tgt_dir / self.inp_tgt_map(img_name)
        assert out_path.exists(), f"For {inp_path=}, target image {out_path} does not exist."
        img_out = util.read_img(None, str(out_path), None)
        H, W, _ = img_out.shape
        if H < 256 or W < 256:
            img_out = cv2.resize(img_out, (256, 256), interpolation=cv2.INTER_CUBIC)
        else: 
            img_out = cv2.resize(img_out, (256, 256), interpolation=cv2.INTER_AREA)
        
        prompt_inp = self.prompt_inp_img.copy()
        prompt_out = self.prompt_out_img.copy()

        prompt_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_inp, (2, 0, 1)))).float()
        prompt_out = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_out, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()
        img_out = torch.from_numpy(np.ascontiguousarray(np.transpose(img_out, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': prompt_inp, 'target_img1': prompt_out,
                 'input_query_img2': img_inp, 'target_img2': img_out,
                 'input_query_img2_path': inp_path}
        return batch, self.task_info
    
    
class DatasetPrompt_InsFilter(Dataset):  # by zkw, adapted from DatasetPrompt_Customize_Prompt
    """
    Dataset for instagram filter addition / removal

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_ori_path, dataset_enh_path, task):
        self.ori_dir = Path(dataset_ori_path)
        self.ori_lst = list(self.ori_dir.iterdir())
        self.enh_dir = Path(dataset_enh_path)
        self.enh_lst = list(self.enh_dir.iterdir())
        
        prompt_ori_path = self.ori_dir / '0_Original.jpg' 
        self.prompt_ori_img = util.read_img(None, str(prompt_ori_path), None)
        prompt_enh_img_dict = {}
        for filter_path in self.enh_lst:
            filter_name = Path(filter_path).stem.split('_')[1]
            if filter_name not in prompt_enh_img_dict:
                prompt_enh_path = self.enh_dir / f'0_{filter_name}.jpg'
                prompt_enh_img_dict[filter_name] = util.read_img(None, str(prompt_enh_path), None)
        self.prompt_enh_img_dict = prompt_enh_img_dict
        
        self.task_info = task
            
    def __len__(self):
        return len(self.enh_lst)

    def __getitem__(self, idx):
        enh_path = self.enh_lst[idx]
        img_enh = util.read_img(None, str(enh_path), None)
        
        img_idx = enh_path.stem.split('_')[0]
        assert 0 <= int(img_idx) < 100, img_idx
        ori_path = self.ori_dir / f'{img_idx}_Original.jpg'
        img_ori = util.read_img(None, str(ori_path), None)
        
        filter_name = enh_path.stem.split('_')[1]
        prompt_enh = self.prompt_enh_img_dict[filter_name].copy()
        prompt_ori = self.prompt_ori_img.copy()
        
        if self.task_info == 'FilterAddition-':
            prompt_inp = prompt_ori
            prompt_out = prompt_enh
            img_inp = img_ori
            img_out = img_enh
        elif self.task_info == 'FilterRemoval-':
            prompt_inp = prompt_enh
            prompt_out = prompt_ori
            img_inp = img_enh
            img_out = img_ori
        else:
            raise ValueError(f"Unknown task: {self.task_info}")

        prompt_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_inp, (2, 0, 1)))).float()
        prompt_out = torch.from_numpy(np.ascontiguousarray(np.transpose(prompt_out, (2, 0, 1)))).float()
        img_inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_inp, (2, 0, 1)))).float()
        img_out = torch.from_numpy(np.ascontiguousarray(np.transpose(img_out, (2, 0, 1)))).float()
        
        batch = {'input_query_img1': prompt_inp, 'target_img1': prompt_out,
                 'input_query_img2': img_inp, 'target_img2': img_out,
                 'input_query_img2_path': str(enh_path)  # always use enh_path to ensure identity
                 }
        return batch, self.task_info