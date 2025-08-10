import os
import cv2
import os.path as osp

import shutil
import random
import numpy as np

import util
from image_operators import adjust_contrast

# MITFivek_data = 'data/MIT-fivek/expert_C_test'
# output_path1 = 'data/Histo_Equ/test/input'
# output_path2 = 'data/Histo_Equ/test/output'
# output_path2 = 'data/MIT-fivek/expert_C_Histogram_Equalization_GT_test'

# imgs_paths, _ = util.get_image_paths('img', MITFivek_data)


# UIEB_data = 'data/UIEB_dive/raw_resize/test'
# output_path1 = 'data/UIEB_Dataset/test_resize'
# output_path2 = 'data/UIEB_Dataset/test_resize_histogram_equalization'

# UIEB_data = 'data/Processd_Datasets/Enhancement/ColorCorrect/Train_256/input'
# output_path1 = 'data/Processd_Datasets/Enhancement/HistoEqual/Train_256/input'
# output_path2 = 'data/Processd_Datasets/Enhancement/HistoEqual/Train_256/target'

UIEB_data = 'data/Processd_Datasets/Enhancement/ColorCorrect/Test_256/input'
output_path1 = 'data/Processd_Datasets/Enhancement/HistoEqual/Test_256/input'
output_path2 = 'data/Processd_Datasets/Enhancement/HistoEqual/Test_256/target'

if not osp.exists(output_path1):
    os.makedirs(output_path1)
if not osp.exists(output_path2):
    os.makedirs(output_path2)
imgs_paths, _ = util.get_image_paths('img', UIEB_data)
random.seed(0)
random.shuffle(imgs_paths)

# for i in range(100):
for i in range(len(imgs_paths)):
    img_path = imgs_paths[i]
    img_name = osp.basename(img_path)
    print(img_name)
    # shutil.copy(img_path, osp.join(output_path1, img_name))
    
    # inp_img = cv2.imread(img_path) / 255.
    # out_img = (adjust_contrast(inp_img) * 255).clip(0, 255).astype(np.uint8)
    
    # inp_img = cv2.imread(img_path)
    # img_resize = cv2.resize(inp_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite(osp.join(output_path1, img_name), img_resize)
    
    # img_resize = img_resize / 255.
    # out_img = (adjust_contrast(img_resize) * 255).clip(0, 255).astype(np.uint8)
    # cv2.imwrite(osp.join(output_path2, img_name), out_img)
    # print(osp.join(output_path2, img_name))
    
    shutil.copy(img_path, osp.join(output_path1, img_name))
    inp_img = cv2.imread(img_path) / 255.

    out_img = (adjust_contrast(inp_img) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(osp.join(output_path2, img_name), out_img)
    print(osp.join(output_path2, img_name))