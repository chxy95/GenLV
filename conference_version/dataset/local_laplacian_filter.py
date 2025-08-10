import os
import cv2
import numpy as np

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def child_window(parent):
    child = parent.copy()
    for K in range(child.shape[0]):
        child[K] = (child[K] + 1) / 2
        child[0] = np.ceil(child[0])
        child[1] = np.floor(child[1])
        child[2] = np.ceil(child[2])
        child[3] = np.floor(child[3])
    return child

def upsample(img, filter, subwindow):
    upsampled = cv2.pyrUp(img)
    upsampled = upsampled[subwindow[0]-1:subwindow[1], subwindow[2]-1:subwindow[3]]
    return upsampled

def pyramid_filter():
    filter = np.array([[1, 4, 6, 4, 1]]) / 16.0
    filter = filter.T.dot(filter)
    return filter

def build_gaussian_pyramid(img, num_levels):
    # build gaussian pyramid
    gaussian_pyramid = [img]
    for level in range(1, num_levels):
        gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[level-1]))
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid, num_levels):
    # build laplacian pyramid
    laplacian_pyramid = [gaussian_pyramid[0]]
    for level in range(1, num_levels):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[level])
        laplacian_pyramid.append(gaussian_pyramid[level-1] - gaussian_expanded)
    return laplacian_pyramid

def reconstruct_laplacian_pyramid(pyr, subwindow=None):
    r = pyr[0].shape[0]
    c = pyr[0].shape[1]
    nlev = len(pyr)

    subwindow_all = np.zeros((nlev, 4))
    if subwindow is None:
        subwindow_all[0, :] = [1, r, 1, c]
    else:
        subwindow_all[0, :] = subwindow
    for lev in range(1, nlev):
        subwindow_all[lev, :] = child_window(subwindow_all[lev-1, :])

    R = pyr[nlev-1]
    filter = pyramid_filter()
    for lev in range(nlev-2, -1, -1):
        R = pyr[lev] + upsample(R, filter, subwindow_all[lev, :])

    return R

def local_laplacian_filter(I, sigma, fact, N):
    height, width, _ = I.shape
    n_levels = int(np.ceil(np.log(min(height, width)) - np.log(2)) + 2)
    discretisation = np.linspace(0, 1, N)
    discretisation_step = discretisation[1]

    input_gaussian_pyr = build_gaussian_pyramid(I, n_levels)
    output_laplace_pyr = build_laplacian_pyramid(input_gaussian_pyr, n_levels)
    output_laplace_pyr[n_levels-1] = input_gaussian_pyr[n_levels-1]

    for ref in discretisation:
        I_remap = fact * (I - ref) * np.exp(-((I - ref) ** 2) / (2 * sigma * sigma))
        temp_gaussian_pyr = build_gaussian_pyramid(I_remap, n_levels)
        temp_laplace_pyr = build_laplacian_pyramid(temp_gaussian_pyr, n_levels)
        for level in range(n_levels-1):
            output_laplace_pyr[level] += (np.abs(input_gaussian_pyr[level] - ref) < discretisation_step) * \
                                         temp_laplace_pyr[level] * \
                                         (1 - np.abs(input_gaussian_pyr[level] - ref) / discretisation_step)

    F = reconstruct_laplacian_pyramid(output_laplace_pyr)

    return F
    

if __name__ == "__main__":
    # 读取输入图像
    img = cv2.imread('/cpfs01/user/puyuandong/glv/Real-ESRGAN/datasets/BSRGAN_degradation/urban100_1/hq/0.png')
    img = uint2single(img)

    # 设置参数
    sigma = 0.1
    fact = 1.0
    N = 256

    # 调用本地拉普拉斯滤波器函数
    filtered_image = local_laplacian_filter(img, sigma, fact, N)
    img = single2uint(filtered_image)
    
    # 保存图像
    cv2.imwrite('example/local_laplacian_filter.png', img)