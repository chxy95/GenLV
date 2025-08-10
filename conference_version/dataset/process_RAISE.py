import os
import cv2

# RAISE_data = 'data/RAISE_1k/original'
# output_path = 'data/RAISE_1k/resize'

# for path in os.listdir(RAISE_data):
#     img_name = osp.basename(path)
#     inp_img = cv2.imread(path, -1)
    
#     img_resize = cv2.resize(inp_img, (256, 256), interpolation=cv2.INTER_LINEAR)
#     cv2.imwrite(osp.join(output_path, img_name), img_resize)

img_path = 'data/RAISE/r0a2e85f0t.TIF'
img = cv2.imread(img_path, -1)
cv2.imwrite('test.png', img)