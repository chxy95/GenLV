# this script is modified from https://github.com/ceciliavision/perceptual-reflection-removal/blob/master/main.py
import os
import cv2
import numpy as np
import scipy.stats as st
import argparse

kernlen = 560
nsig = 3
interval = (2*nsig+1.)/(kernlen)
x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
kern1d = np.diff(st.norm.cdf(x))
kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
g_mask = kernel_raw/kernel_raw.sum()
g_mask = g_mask/g_mask.max()
g_mask = np.dstack((g_mask, g_mask, g_mask))
k_sz = np.linspace(1, 5, 80)
sigma = k_sz[np.random.randint(0, len(k_sz))]
def syn_data(t, r, sigma):
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2*np.ceil(2*sigma)+1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur+t

    att = 1.08+np.random.random()/10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i]*maski)/(maski.sum()+1e-6))
        r_blur[:, :, i] = r_blur[:, :, i]-(mean_i-1)*att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    neww = np.random.randint(0, 560-w-10)
    newh = np.random.randint(0, 560-h-10)
    alpha1 = g_mask[newh:newh+h, neww:neww+w, :]
    alpha2 = 1-np.random.random()/5.0
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask+t*alpha2

    t = np.power(t, 1/2.2)
    r_blur_mask = np.power(r_blur_mask, 1/2.2)
    blend = np.power(blend, 1/2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend

def add_reflection(t, r):
    # create a vignetting mask, a 2D Gaussian kernel array.
    kernlen = 560
    nsig = 3
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    g_mask = kernel_raw/kernel_raw.sum()
    g_mask = g_mask/g_mask.max()
    g_mask = np.dstack((g_mask, g_mask, g_mask))

    k_sz = np.linspace(1, 5, 80)
    sigma = k_sz[np.random.randint(0, len(k_sz))]
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2*np.ceil(2*sigma)+1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur+t

    att = 1.08+np.random.random()/10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i]*maski)/(maski.sum()+1e-6))
        r_blur[:, :, i] = r_blur[:, :, i]-(mean_i-1)*att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    neww = np.random.randint(0, 560-w-10)
    newh = np.random.randint(0, 560-h-10)
    alpha1 = g_mask[newh:newh+h, neww:neww+w, :]
    alpha2 = 1-np.random.random()/5.0
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask+t*alpha2

    t = np.power(t, 1/2.2)
    r_blur_mask = np.power(r_blur_mask, 1/2.2)
    blend = np.power(blend, 1/2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def save_image(image, path):
    """
    image: saved image, numpy array, dtype=float
    path: saving path
    """
    # The type of the image is float, and range of the image might not be in [0, 255]
    # Thus, before saving the image, the image needs to be clipped.
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)

    # save image
    cv2.imwrite(path, image)

def read_img(env, path, size=None, float=True):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if float:
        img = img.astype(np.float32) / 255.
    return img

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/Restoration/reflection_removal/zhang/synthetic/transmission_layer/5.jpg')
    parser.add_argument('--output_path', type=str, default='example')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":
    
    opt = init_parser()
    os.makedirs(opt.output_path, exist_ok=True)
    # img = cv2.imread(opt.input_path, -1)
    # img = cv2.resize(img, (256, 256))
    # img = cv2.resize(np.float32(img), (256, 256))/255.0
    # img_reflection = cv2.imread('data/Restoration/reflection_removal/zhang/synthetic/reflection_layer/5.jpg', -1)
    # img_reflection = cv2.resize(img_reflection, (256, 256))
    # img_reflection = cv2.resize(np.float32(img_reflection), (256, 256))/255.0
    # img = uint2single(img)
    # img_reflective = uint2single(img_reflection)
    img = read_img(None, opt.input_path)
    img_reflection = read_img(None, 'data/Restoration/reflection_removal/zhang/synthetic/reflection_layer/5.jpg')
    img = cv2.resize(img, (256, 256))
    img_reflection = cv2.resize(img_reflection, (256, 256))
    _, _, degraded_img = add_reflection(img, img_reflection)
    degraded_img = single2uint(degraded_img)
    save_image(degraded_img, os.path.join(opt.output_path, "add_reflection_old.png"))