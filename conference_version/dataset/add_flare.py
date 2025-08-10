import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import argparse

import numpy as np
from PIL import Image
import glob
import random
import cv2

from scipy import ndimage
import skimage
from skimage import morphology
from skimage.measure import label
from skimage.filters import rank
from skimage.morphology import disk
from skimage import color
from skimage.measure import regionprops

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch

def plot_light_pos(input_img,threshold):
	#input should be a three channel tensor with shape [C,H,W]
	#Out put the position (x,y) in int
 
	luminance=0.3*input_img[0]+0.59*input_img[1]+0.11*input_img[2]
	luminance_mask=luminance>threshold
	luminance_mask_np=luminance_mask.numpy()
	struc = disk(3)
	img_e = ndimage.binary_erosion(luminance_mask_np, structure = struc)
	img_ed = ndimage.binary_dilation(img_e, structure = struc)

	labels = label(img_ed)
	if labels.max() == 0:
		print("Light source not found.")
		return (255,255)
	else:
         largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
         largestCC=largestCC.astype(int)
         properties = regionprops(largestCC, largestCC)
        #  properties = regionprops(largestCC, luminance)
         weighted_center_of_mass = properties[0].weighted_centroid
         print("Light source detected in position: x:",int(weighted_center_of_mass[1]),",y:",int(weighted_center_of_mass[0]))
         return (int(weighted_center_of_mass[1]),int(weighted_center_of_mass[0]))

class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)

class TranslationTransform(object):
    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x,angle=0, scale=1,shear=[0,0], translate= list(self.position))
    
def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image

def addflare(x, flare_list):
    
    transform_base=transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect')
							#   transforms.RandomHorizontalFlip()
                              ])
    
    # load base image
    base_img = x
    # 确保图像是RGB
    base_img = base_img.convert('RGB')
    gamma=np.random.uniform(1.8,2.2)
    to_tensor=transforms.ToTensor()
    adjust_gamma=RandomGammaCorrection(gamma)
    adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
    color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
    base_img=to_tensor(base_img)
    base_img=adjust_gamma(base_img)
    base_img=transform_base(base_img)
    sigma_chi=0.01*np.random.chisquare(df=1)
    base_img=Normal(base_img,sigma_chi).sample()
    gain=np.random.uniform(1,1.2)
    flare_DC_offset=np.random.uniform(-0.02,0.02)
    base_img=gain*base_img
    base_img=torch.clamp(base_img,min=0,max=1)

    light_pos=plot_light_pos(base_img,0.97**gamma)
    light_pos=[light_pos[0]-256,light_pos[1]-256]
    
    #traslate=TranslationTransform(light_pos)
    transform_flare=transforms.Compose(
                            [transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(0,0),shear=(-20,20)),
                            TranslationTransform(light_pos),
                            transforms.CenterCrop((512,512)),
                            ])

    ##load flare image
    flare_path=random.choice(flare_list)
    flare_img =Image.open(flare_path)

    flare_img=to_tensor(flare_img)
    flare_img=adjust_gamma(flare_img)
    flare_img=remove_background(flare_img)
    flare_img=transform_flare(flare_img)
    
    #change color
    flare_img=color_jitter(flare_img)

    #flare blur
    blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
    flare_img=blur_transform(flare_img)
    flare_img=flare_img+flare_DC_offset
    flare_img=torch.clamp(flare_img,min=0,max=1)

    #merge image	
    merge_img=flare_img+base_img
    merge_img=torch.clamp(merge_img,min=0,max=1)
    merge_img = adjust_gamma_reverse(merge_img)
    
    return merge_img

def add_flare(img, img_flare=None, img_reflective=None, img_light=None):
    # written by YuandongPu
    # 需要保证四个图像的尺寸一致
    img_H, img_W, _ = img.shape
    transform_base = transforms.Compose([transforms.RandomCrop((512, 512),pad_if_needed=True,padding_mode='reflect')
							#   transforms.RandomHorizontalFlip()
                              ])
    
    
    # load base image
    base_img = img.copy()
    
    gamma = np.random.uniform(1.8,2.2)
    to_tensor = transforms.ToTensor()
    adjust_gamma = RandomGammaCorrection(gamma)
    adjust_gamma_reverse = RandomGammaCorrection(1/gamma)
    color_jitter = transforms.ColorJitter(brightness = (0.8,3),hue = 0.0)
    
    base_img = to_tensor(base_img)
    base_img = adjust_gamma(base_img)
    base_img = transform_base(base_img)
    
    sigma_chi = 0.01 * np.random.chisquare(df = 1)
    base_img = Normal(base_img,sigma_chi).sample()
    gain = np.random.uniform(1,1.2)
    flare_DC_offset = np.random.uniform(-0.02,0.02)
    base_img = gain*base_img
    base_img = torch.clamp(base_img,min = 0,max = 1)

    light_pos = plot_light_pos(base_img,0.97**gamma)
    light_pos = [light_pos[0]-256,light_pos[1]-256]
    
    #traslate = TranslationTransform(light_pos)
    transform_flare = transforms.Compose(
                            [transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees = (0,360),scale = (0.8,1.5),translate = (0,0),shear = (-20,20)),
                            TranslationTransform(light_pos),
                            transforms.CenterCrop((512, 512)),
                            ])

    ##load flare image
    img_flare = to_tensor(img_flare)
    img_flare = adjust_gamma(img_flare)
    
    if img_reflective is not None:
        #image flare的形状和image reflective的形状不一致，需要想办法解决
        img_reflective = to_tensor(img_reflective)
        img_reflective = adjust_gamma(img_reflective)
        img_flare = torch.clamp(img_flare+img_reflective,min=0,max=1)
        
    img_flare = remove_background(img_flare)
    
    if img_light is not None:
        img_light = to_tensor(img_light)
        img_light = adjust_gamma(img_light)
        flare_merge=torch.cat((img_flare, img_light), dim=0)
        flare_merge=transform_flare(flare_merge)
        img_flare, img_light = torch.split(flare_merge, 3, dim=0)
    else:
        img_flare = transform_flare(img_flare)
        
        #change color
        img_flare = color_jitter(img_flare)

    #flare blur
    blur_transform = transforms.GaussianBlur(21,sigma = (0.1,3.0))
    img_flare = blur_transform(img_flare)
    img_flare = img_flare+flare_DC_offset
    img_flare = torch.clamp(img_flare,min = 0,max = 1)

    #merge image	
    merge_img = img_flare+base_img
    merge_img = torch.clamp(merge_img,min = 0,max = 1)
    merge_img = adjust_gamma_reverse(merge_img)
    merge_img = merge_img.permute(1,2,0).numpy()
    
    if img_light is not None:
        base_img=base_img+img_light
        base_img=torch.clamp(base_img,min=0,max=1)
        img_flare=img_flare-img_light
        img_flare=torch.clamp(img_flare,min=0,max=1)
    
    base_img = adjust_gamma_reverse(base_img)
    base_img = base_img.permute(1,2,0).numpy()

    return merge_img, base_img

def get_highlight_mask(im, threshold=0.99, dtype=torch.float32):
    """Returns a binary mask indicating the saturated regions in the input image.

    Args:
        im: Image tensor with shape [H, W, C] or [B, C, H, W].
        threshold: A pixel is considered saturated if its channel-averaged intensity
            is above this value.
        dtype: Expected output data type.

    Returns:
        A `dtype` tensor with shape [H, W, 1] or [B, 1, H, W].
    """
    # channel_avg = torch.mean(im, dim=-3, keepdim=True)
    channel_avg = np.mean(im, axis=-1, keepdims=True)
    binary_mask = channel_avg > threshold
    # convert to bool
    mask = binary_mask.astype(bool)
    return mask


def refine_mask(mask, morph_size=0.01):
    """Refines a mask by applying morphological operations.

    Args:
        mask: A float array of shape [H, W] or [B, H, W].
        morph_size: Size of the morphological kernel relative to the long side of
            the image.

    Returns:
        Refined mask of shape [H, W] or [B, H, W].
    """
    mask_size = max(mask.shape[-2:])
    kernel_radius = 0.5 * morph_size * mask_size
    kernel = skimage.morphology.disk(int(np.ceil(kernel_radius)))
    # opened = torch.tensor(skimage.morphology.binary_opening(mask.numpy(), kernel))
    opened = skimage.morphology.binary_opening(mask, kernel)
    return opened

def _create_disk_kernel(kernel_size):
    _EPS = 1e-7
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2 + yy**2)
    kernel = np.float32(rr <= np.max(x)) + _EPS
    kernel = kernel / np.sum(kernel)
    return kernel

def blend_light_source(scene_input, scene_pred):
    """Adds suspected light source in the input to the flare-free image.
        scene_input: with flare
        scene_pred: without flare
    """
    #   binary_mask = get_highlight_mask(scene_input, dtype=torch.bool).numpy()
    binary_mask = get_highlight_mask(scene_input)
    binary_mask = np.squeeze(binary_mask, axis=-1)
    binary_mask = refine_mask(binary_mask)

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)
    max_diameter = 0
    for p in properties:
        max_diameter = max(max_diameter, p['equivalent_diameter'])

    mask = np.float32(binary_mask)

    kernel_size = round(1.5 * max_diameter)
    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask * 3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=-1)
    else:
        mask_rgb = 0

    blend = scene_input * mask_rgb + scene_pred * (1 - mask_rgb)

    return blend

def add_lens_flare(img, img_flare=None):
    # written by YuandongPu
    # merge_image:lq
    # blend_image:gt
    img_H, img_W, _ = img.shape
    transform_base = transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect')
							#   transforms.RandomHorizontalFlip()
                              ])
    
    
    # load base image
    base_img = img.copy()
    
    gamma = np.random.uniform(1.8,2.2)
    to_tensor = transforms.ToTensor()
    adjust_gamma = RandomGammaCorrection(gamma)
    adjust_gamma_reverse = RandomGammaCorrection(1/gamma)
    color_jitter = transforms.ColorJitter(brightness = (0.8,3),hue = 0.0)
    
    base_img = to_tensor(base_img)
    base_img = adjust_gamma(base_img)
    base_img = transform_base(base_img)
    
    sigma_chi = 0.01 * np.random.chisquare(df = 1)
    base_img = Normal(base_img,sigma_chi).sample()
    gain = np.random.uniform(1,1.2)
    flare_DC_offset = np.random.uniform(-0.02,0.02)
    base_img = gain*base_img
    base_img = torch.clamp(base_img,min = 0,max = 1)

    light_pos = plot_light_pos(base_img,0.97**gamma)
    light_pos = [light_pos[0]-256,light_pos[1]-256]
    
    #traslate = TranslationTransform(light_pos)
    transform_flare = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.RandomAffine(degrees = (0,360),scale = (0.8,1.5),translate = (0,0),shear = (-20,20)),
                            # TranslationTransform(light_pos),
                            # transforms.CenterCrop((512, 512)),
                            transforms.Resize((512,512)),
                            # transforms.CenterCrop((img_H,img_W)),
                            ])

    ##load flare image
    img_flare = to_tensor(img_flare)
    img_flare = adjust_gamma(img_flare)
    
        
    img_flare = remove_background(img_flare)
    img_flare = transform_flare(img_flare)
    # if light_source_flag == False:
    #     #save flare img
    #     flare_img = img_flare.permute(1,2,0).numpy()
    #     flare_img = single2uint(flare_img)
    #     cv2.imwrite('flare_img.png', flare_img)
    #change color
    img_flare = color_jitter(img_flare)

    #flare blur
    blur_transform = transforms.GaussianBlur(21,sigma = (0.1,3.0))
    img_flare = blur_transform(img_flare)
    img_flare = img_flare+flare_DC_offset
    img_flare = torch.clamp(img_flare,min = 0,max = 1)

    #merge image	
    merge_img = img_flare+base_img
    merge_img = torch.clamp(merge_img,min = 0,max = 1)
    merge_img = adjust_gamma_reverse(merge_img)
    merge_img = merge_img.permute(1,2,0).numpy()
    
    #generate flare mask
    
    base_img = adjust_gamma_reverse(base_img)
    base_img = base_img.permute(1,2,0).numpy()

    blend_img = blend_light_source(merge_img, base_img)
    return merge_img, blend_img

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


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/Restoration/Flare7Kpp/Flickr24K/5.jpg')
    parser.add_argument('--output_path', type=str, default='example')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    
    opt = init_parser()
    os.makedirs(opt.output_path, exist_ok=True)
    img = cv2.imread(opt.input_path)
    # img_flare = cv2.imread('data/Restoration/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare/000001.png')
    # img_reflective = cv2.imread('data/Restoration/Flare7Kpp/Flare7K/Reflective_Flare/000000.png')
    # img_light = cv2.imread('data/Restoration/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source/000000.png')
    
    img_flare = cv2.imread('data/Restoration/lens-flare/simulated/aperture0019_blur03_crop02.png')
    img = uint2single(img)
    img_flare = uint2single(img_flare)
    # img_reflective = uint2single(img_reflective)
    # img_light = uint2single(img_light)
    degraded_img, blend_img = add_lens_flare(img, img_flare)
    degraded_img = single2uint(degraded_img)
    blend_img = single2uint(blend_img)
    save_image(degraded_img, os.path.join(opt.output_path, "add_lens_flare.png"))
    save_image(blend_img, os.path.join(opt.output_path, "blend_img.png"))