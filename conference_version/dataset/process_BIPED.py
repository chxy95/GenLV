import os
import os.path as osp

import cv2
import util

import numpy as np

BIPED_data = 'data/BIPED/v2/edges/edge_maps/test/rgbr'
output_path = 'data/BIPED/resize/edges/test'

imgs_paths, _ = util.get_image_paths('img', BIPED_data)

for img_path in imgs_paths:
    img_name = osp.basename(img_path)
    inp_img = cv2.imread(img_path, -1)
    # print(inp_img.shape)
    
    img_resize = cv2.resize(inp_img, (256, 256), interpolation=cv2.INTER_AREA)
    # kernel = np.ones((3,3), np.uint8)
    # processed_edges = cv2.dilate(img_resize, kernel, iterations=1)
    cv2.imwrite(osp.join(output_path, img_name), img_resize)
    
    