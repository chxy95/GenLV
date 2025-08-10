#Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation
#速度较慢，不建议使用
import os
import cv2
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def sigmoid(x, a):
    # Apply Sigmoid
    y = 1 / (1 + np.exp(-a * x))

    # Re-scale
    y05 = 1 / (1 + np.exp(-a * 0.5))
    y = y * (0.5 / y05) - 0.5

    return y

def zeroone(in_data):
    max_in = np.max(in_data)
    min_in = np.min(in_data)

    out = (in_data - min_in) / (max_in - min_in)
    return out

def wlsFilter(IN, lambda_val=1.0, alpha=1.2, L=None):
    if L is None:
        L = np.log(IN + np.finfo(float).eps)
    
    smallNum = 0.0001
    
    r, c = IN.shape
    k = r * c
    
    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, axis=0)
    dy = -lambda_val / (np.abs(dy) ** alpha + smallNum)
    dy = np.pad(dy, [(0, 1), (0, 0)], 'constant', constant_values=0)
    dy = dy.ravel()
    
    dx = np.diff(L, axis=1)
    dx = -lambda_val / (np.abs(dx) ** alpha + smallNum)
    dx = np.pad(dx, [(0, 0), (0, 1)], 'constant', constant_values=0)
    dx = dx.ravel()
    
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    B = np.column_stack((dx, dy))
    d = [-r, -1]
    A = spdiags(B.T, d, k, k)
    
    e = dx
    w = np.pad(dx, (r, 0), 'constant', constant_values=0)[:-r]
    s = dy
    n = np.pad(dy, (1, 0), 'constant', constant_values=0)[:-1]
    
    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)
    
    # Solve
    # OUT = np.linalg.solve(A.todense(), IN.ravel())
    A = A.tocsc()
    # linalg.cg(A, IN.ravel())
    OUT = spsolve(A, IN.ravel())
    OUT = np.reshape(OUT, (r, c))
    
    return OUT

# def wlsFilter(img, lambda_val, alpha_val):
#     wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lambda_val, alpha_val)
#     filtered_img = wls_filter.filter(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
#     return cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

def tonemapLAB(lab, L0, L1, val0, val1, val2, exposure, gamma, saturation):
    L = lab[:,:,0]

    if val0 == 0:
        diff0 = L - L0
    else:
        if val0 > 0:
            diff0 = sigmoid((L - L0) / 100, val0) * 100
        else:
            diff0 = (1 + val0) * (L - L0)

    if val1 == 0:
        diff1 = L0 - L1
    else:
        if val1 > 0:
            diff1 = sigmoid((L0 - L1) / 100, val1) * 100
        else:
            diff1 = (1 + val1) * (L0 - L1)

    if val2 == 0:
        base = exposure * L1
    else:
        if val2 > 0:
            base = (sigmoid((exposure * L1 - 56) / 100, val2) * 100) + 56
        else:
            base = (1 + val2) * (exposure * L1 - 56) + 56

    if gamma == 1:
        res = base + diff1 + diff0
    else:
        maxBase = np.max(base)
        res = (zeroone(base) ** gamma) * maxBase + diff1 + diff0

    if saturation == 0:
        lab[:,:,0] = res
    else:
        lab[:,:,0] = res
        lab[:,:,1] *= saturation
        lab[:,:,2] *= saturation

    cform = cv2.COLOR_Lab2BGR
    res = cv2.cvtColor(lab, cform)
    return res

def multi_scale_detail_manipulation(img):
    # Load image
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Filter
    L0 = wlsFilter(L, 0.125, 1.2)
    L1 = wlsFilter(L, 0.50, 1.2)

    # Fine
    val0 = 25
    val1 = 1
    val2 = 1
    exposure = 1.0
    saturation = 1.1
    gamma = 1.0
    
    fine = tonemapLAB(lab, L0, L1, val0, val1, val2, exposure, gamma, saturation)

    # Medium
    val0 = 1
    val1 = 40
    val2 = 1
    exposure = 1.0
    saturation = 1.1
    gamma = 1.0

    med = tonemapLAB(lab, L0, L1, val0, val1, val2, exposure, gamma, saturation)

    # Coarse
    val0 = 4
    val1 = 1
    val2 = 15
    exposure = 1.10
    saturation = 1.1
    gamma = 1.0

    coarse = tonemapLAB(lab, L0, L1, val0, val1, val2, exposure, gamma, saturation)
    
    return fine, med, coarse

def progressive_image_abstraction():
    pass

if __name__ == "__main__":
    img = cv2.imread('/cpfs01/user/puyuandong/glv/Real-ESRGAN/datasets/BSRGAN_degradation/urban100_1/hq/0.png')
    img = uint2single(img)
    fine, medium, corse = multi_scale_detail_manipulation(img)
    fine = single2uint(fine)
    medium = single2uint(medium)
    corse = single2uint(corse)
    cv2.imwrite('example/fine.png', fine)
    cv2.imwrite('example/medium.png', medium)
    cv2.imwrite('example/corse.png', corse)
