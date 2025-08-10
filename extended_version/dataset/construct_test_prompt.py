import os
import os.path as osp

import cv2
import numpy as np

import util
from add_degradation_various import *
from image_operators import *
from x_distortion import *
from basicsr.utils.matlab_functions import imresize

def calculate_operators_one_image(img_inp, deg_type):
    # np.uint8
    if deg_type == 'Laplacian':
        img_out = Laplacian_edge_detector_uint8(img_inp)
    elif deg_type == 'Canny':
        img_out = Canny_edge_detector_uint8(img_inp)
    return img_out, img_inp 

def add_degradation_one_image(img_gt, deg_type):
    # np.float32
    if deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq = add_rain(img_gt, value=value)
    elif deg_type == 'Ringing':
        img_lq = add_ringing(img_gt)
    elif deg_type == 'r_l':
        img_lq = r_l(img_gt)
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_lq = inpainting(img_gt, l_num=l_num, l_thick=l_thick)
    elif deg_type == 'mosaic':
        img_lq = mosaic_CFA_Bayer(img_gt)
    elif deg_type == 'SRx2':
        H, W, _ = img_gt.shape
        img_lq = imresize(img_gt, 1/2)
        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)
    elif deg_type == 'SRx4':
        H, W, _ = img_gt.shape
        img_lq = imresize(img_gt, 1/4)
        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

    elif deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq = add_Gaussian_noise(img_gt, level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq = iso_GaussianBlur(img_gt, window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq = add_JPEG_noise(img_gt, level=level)
    elif deg_type == 'Resize':
        img_lq = add_resize(img_gt)
    elif deg_type == 'SPNoise':
        img_lq = add_sp_noise(img_gt)
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq = low_light(img_gt, lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq = add_Poisson_noise(img_gt, level=2)
    elif deg_type == 'gray':
        img_lq = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        img_lq = np.expand_dims(img_lq, axis=2)
        img_lq = np.concatenate((img_lq, img_lq, img_lq), axis=2)
    elif deg_type == 'None':
        img_lq = img_gt
    else:
        print('Error!', '-', deg_type, '-')
        exit()

    img_lq = np.clip(img_lq*255, 0, 255).round().astype(np.uint8)
    img_gt = np.clip(img_gt*255, 0, 255).round().astype(np.uint8)
    
    return img_lq, img_gt

img_save_path = '../data/Common528_plus/prompts'
os.makedirs(img_save_path, exist_ok=True)
prompt_path = '../data/Common528_plus/target_256x256/046.png'

# -------------------- onthefly_degradation_list1 --------------------
onthefly_degradation_list1 = ['Rain', 'Ringing', 'r_l', 'Inpainting', 'mosaic', 'SRx2', 'SRx4']
# for deg_type in onthefly_degradation_list1:
#     img = util.read_img(None, prompt_path, None, float=True)
#     img_out, _ = add_degradation_one_image(img, deg_type)
#     out_path = osp.join(img_save_path, deg_type+'.png')
#     cv2.imwrite(out_path, img_out)

# -------------------- onthefly_degradation_list2 --------------------
onthefly_degradation_list2 = ['Laplacian', 'Canny']
# for deg_type in onthefly_degradation_list2:
#     img = util.read_img(None, prompt_path, None, float=False)
#     img_out, _ = calculate_operators_one_image(img, deg_type)
#     out_path = osp.join(img_save_path, deg_type+'.png')
#     cv2.imwrite(out_path, img_out)

# -------------------- onthefly_degradation_list3 --------------------
# def add_x_distortion_1(img_gt, deg_type, severity):
#     # np.uint8, BGR
#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
#     img_lq = globals()[deg_type](img_gt, severity)

#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
#     img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
    
#     return img_lq, img_gt

onthefly_degradation_list3 = ['blur_gaussian', 'blur_motion', 'blur_glass', 'blur_lens', 'blur_zoom', 'blur_jitter', 
                              'noise_speckle', 'noise_spatially_correlated', 'noise_poisson', 'noise_impulse', 
                              'compression_jpeg', 'compression_jpeg_2000', 'oversharpen', 'pixelate', 
                              'quantization_otsu', 'quantization_median', 'quantization_hist', 'spatter']

# for deg_type in onthefly_degradation_list3:
#     img = util.read_img(None, prompt_path, None, float=False)
#     for severity in [1, 2, 3, 4, 5]:
#         img_out, _ = add_x_distortion_1(img, deg_type, severity)
#         out_path = osp.join(img_save_path, '_'.join([deg_type, str(severity)])+ '.png')
#         cv2.imwrite(out_path, img_out)

# -------------------- onthefly_degradation_list4 --------------------
# def add_x_distortion_2(img_gt, deg_type, severity):
#     # np.uint8, BGR
#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
#     img_lq = globals()[deg_type](img_gt, severity)

#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
#     img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
    
#     return img_lq, img_gt, deg_type, severity

onthefly_degradation_list4 = ['noise_gaussian', 'brightness_brighten', 'brightness_darken', 
                              'contrast_strengthen', 'contrast_weaken', 'saturate_strengthen', 'saturate_weaken']

# for task_type in onthefly_degradation_list4:
#     if task_type == 'noise_gaussian':
#         deg_type_list = ['noise_gaussian_RGB', 'noise_gaussian_YCrCb']
#     elif task_type == 'brightness_brighten':
#         deg_type_list = distortions_dict['brighten']
#     elif task_type == 'brightness_darken':
#         deg_type_list = distortions_dict['darken']
#     else: deg_type_list = distortions_dict[task_type]
    
#     img = util.read_img(None, prompt_path, None, float=False)
#     for deg_type in deg_type_list:
#         for severity in [1, 2, 3, 4, 5]:
#             img_out, _, deg_type, severity = add_x_distortion_2(img, deg_type, severity)
#             out_path = osp.join(img_save_path, '_'.join([deg_type, str(severity)])+ '.png')
#             cv2.imwrite(out_path, img_out)

# -------------------- onthefly_OOD_blur_gaussian_lensmask --------------------
# def add_x_distortion(img_gt, deg_type, severity):
#     # np.uint8, BGR
#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
#     img_lq = globals()[deg_type](img_gt, severity)

#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
#     img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
    
#     return img_lq, img_gt

# deg_type = 'blur_gaussian_lensmask'
# img = util.read_img(None, prompt_path, None, float=False)
# for severity in [1, 2, 3, 4, 5]:
#     img_out, _ = add_x_distortion(img, deg_type, severity)
#     out_path = osp.join(img_save_path, '_'.join([deg_type, str(severity)])+ '.png')
#     cv2.imwrite(out_path, img_out)

# -------------------- onthefly_OOD_gaussian_noise --------------------
# def add_ood_gaussian_noise(img, sigma):
#     # np.uint8, BGR
#     noise = np.random.normal(0, sigma, img.shape)
#     img_lq = img + noise
#     return np.uint8(np.clip(img_lq, 0, 1) * 255.), sigma

# deg_type = 'OOD_gaussian_noise'
# img = util.read_img(None, prompt_path, None, float=True)
# for sigma in [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.35, 0.50]:
#     img_out, _ = add_ood_gaussian_noise(img, sigma)
#     out_path = osp.join(img_save_path, '_'.join([deg_type, str(sigma)])+ '.png')
#     cv2.imwrite(out_path, img_out)

# -------------------- onthefly_OOD_blur_gaussian --------------------
def add_ood_blur_gaussian(img, sigma):
    # np.uint8, BGR
    img = np.array(img) / 255.
    img = gaussian(img, sigma=sigma, channel_axis=-1)
    img = np.clip(img, 0, 1) * 255
    return img.round().astype(np.uint8), sigma

deg_type = 'OOD_blur_gaussian'
img = util.read_img(None, prompt_path, None, float=False)
for sigma in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
    img_out, _ = add_ood_blur_gaussian(img, sigma)
    out_path = osp.join(img_save_path, '_'.join([deg_type, str(sigma)])+ '.png')
    cv2.imwrite(out_path, img_out)