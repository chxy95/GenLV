'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-04-29 15:30:09
@LastEditors: xiaoshuyui
@LastEditTime: 2020-05-08 13:48:55
运行速度过慢，需要优化，暂不考虑
https://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/
'''
import cv2
import numpy as np 
import copy
import scipy.sparse as sparse
from scipy.sparse import csr_matrix

def tsmooth(img,lamda=0.01,sigma=3.0,sharpness=0.02,maxIter=4):
    """
    >>> img = cv2.imread("path/of/your-file")
    >>> s = tsmooth(img)
    """
    img = np.array(img,dtype=np.float32)/255

    # print(np.max(img))
    # print(np.min(img))

    x = copy.deepcopy(img)
    sigma_iter = sigma
    lamda = lamda/2.0
    dec = 2.0
    for it in range(0,maxIter):
        wx,wy = computeTextureWeights(x,sigma_iter,sharpness)
        x = solveLinearEquation(img,wx,wy,lamda)
        sigma_iter = sigma_iter/dec
        if sigma_iter<0.5:
            sigma_iter = 0.5
    
    return x




def matCopy(ma):
    # ma1 = copy.deepcopy(ma)
    # ma2 = copy.deepcopy(ma)
    return cv2.merge([ma,ma,ma])  

def matSum(ma):
    b, g, r = cv2.split(ma) 

    return(b+g+r)

def mat3tomat3(fin,finshape):
    b,g,r = cv2.split(fin) 
    resX = []
    resY = []
    for i in [b,g,r]:
        fx = np.diff(i)
        col = np.zeros((finshape[0],1))
        fx = np.concatenate((fx,col),axis=1)
        fy = np.diff(i,axis=0)
        row = np.zeros((1,finshape[1]))
        fy = np.concatenate((fy,row),axis=0)

        resX.append(fx)
        resY.append(fy)
    
    fx = cv2.merge([resX[0],resX[1],resX[2]])
    fy = cv2.merge([resY[0],resY[1],resY[2]])

    return fx,fy



def computeTextureWeights(fin,sigma,sharpness):
    finshape = fin.shape
    vareps_s = sharpness
    vareps = 0.001  
    if len(finshape)==2:
        fx = np.diff(fin)
        col = np.zeros((finshape[0],1))
        fx = np.concatenate((fx,col),axis=1)
        fy = np.diff(fin,axis=0)
        row = np.zeros((1,finshape[1]))
        fy = np.concatenate((fy,row),axis=0)

 
        fx = matCopy(fx)
        fy = matCopy(fy)
    else:
        # b,g,r = cv2.split(fin) 
        fx,fy = mat3tomat3(fin,finshape)


    wto_sum = matSum(np.sqrt(np.square(fx)+np.square(fy)))/3
    wto_sum[wto_sum<sharpness] = sharpness

    wto = np.power(wto_sum,-1)

    fbin = lpfilter(fin,sigma)
    
    gfx,gfy = mat3tomat3(fbin,finshape)

    wtbx_sum = matSum(np.abs(gfx))/3
    wtbx_sum[wtbx_sum<vareps] = vareps
    wtbx = np.power(wtbx_sum,-1)

    wtby_sum = matSum(np.abs(gfy))/3
    wtby_sum[wtby_sum<vareps] = vareps
    wtby = np.power(wtby_sum,-1)

    retx = wtbx * wto
    rety = wtby * wto

    retx[:,finshape[1]-1] = 0
    rety[finshape[0]-1,0] = 0

    return retx,rety


def conv2_sep(im,sigma):
    im = np.array(im,dtype=np.float32)
    ksize = max(round(5*sigma),1)
    # print(ksize)
    if ksize % 2 == 0:
        ksize = ksize +1
    dst = cv2.GaussianBlur(im,(1,ksize),sigma)
    return cv2.GaussianBlur(dst,(1,ksize),sigma)

def test_conv2_sep(im=np.random.randint(255,size=(256,256)),sigma=3.0):
    im = np.array(im,dtype=np.float32)
    cv2.imwrite("./test.jpg",conv2_sep(im,sigma))



def lpfilter(fimg,sigma):
    fbimg = fimg
    b,g,r = cv2.split(fbimg)
    
    b1 = conv2_sep(b,sigma)
    g1 = conv2_sep(g,sigma)
    r1 = conv2_sep(r,sigma)

    return cv2.merge([b1,g1,r1])


def solveLinearEquation(IN,wx:np.ndarray,wy:np.ndarray,lamda):
    r,c,ch = IN.shape
    k = r * c

    dx = -lamda * wx.T.ravel()
    dy = -lamda * wy.T.ravel()

    B = np.vstack([dx,dy]).T

    d = [-r,-1]
    A = sparse.spdiags(B.T,d,k,k)
    # print(A.shape)
    dx = np.reshape(dx,(len(dx),1))
    dy = np.reshape(dy,(len(dy),1))
    e = dx
    tmp = np.zeros((r,1),dtype=np.float32)

    w_ = np.concatenate((tmp,dx),axis=0)
    w = w_[0:len(w_)-r]
    s = dy
    n_ =  np.concatenate((np.array([0]).reshape((1,1)),dy),axis=0)
    n = n_[0:len(n_)-1]
    D = 1-(e+w+s+n)

    A = A + A.T +sparse.spdiags(D.T,0,k,k)

    A = csr_matrix(A,dtype=float)

    A = A.A.astype(np.float32)
    # print("===============>start")
    # print(np.max(A))
    # print(np.min(A))
    # print("===============>end")
    OUT = IN
    outs = []
    for i in range(0,3):
        tin = IN[:,:,i]
        tin_b = tin.T.ravel()
        # tout = np.dot(tin_b,A)
        tout = np.dot(np.linalg.inv(A),tin_b)

        
        tout = np.reshape(tout,(r,c))
        outs.append(tout)
        # print(tout.shape)
        
    # print(outs[0])
    return cv2.merge([outs[0],outs[1],outs[2]])


def test_solveLinearEquation(IN=np.random.randint(255,size=(3,4,5)),wx=np.random.rand(3,4), \
    wy = np.random.rand(3,4),lamda=0.01):
    r,c,ch = IN.shape
    k = r * c

    dx = -lamda * wx.T.ravel()
    dy = -lamda * wy.T.ravel()

    # print(dx.shape)
    # print(dy.shape)

    # dx = np.array(dx)
    # dy = np.array(dy)

    # B = np.concatenate((dx,dy),axis=1)
    B = np.vstack([dx,dy]).T
    # print(B.shape)
    d = [-r,-1]
    A = sparse.spdiags(B.T,d,k,k)
    # print(A.shape)
    dx = np.reshape(dx,(len(dx),1))
    dy = np.reshape(dy,(len(dy),1))
    e = dx
    tmp = np.zeros((r,1),dtype=np.float32)
    # print(tmp.shape)
    

    # print(dx.shape)
    w_ = np.concatenate((tmp,dx),axis=0)
    w = w_[0:len(w_)-r]
    s = dy
    n_ =  np.concatenate((np.array([0]).reshape((1,1)),dy),axis=0)
    n = n_[0:len(n_)-1]

    # print(e.shape)
    # print(w.shape)
    # print(s.shape)
    # print(n.shape)

    D = 1-(e+w+s+n)

    A = A + A.T +sparse.spdiags(D.T,0,k,k)

    A = A.A
    print(type(A))
    OUT = IN
    outs = []
    for i in range(0,3):
        tin = IN[:,:,i]
        tin_b = tin.T.ravel()
        print(tin_b)
        tout = np.dot(tin_b,A)
        
        tout = np.reshape(tout,(r,c))
        outs.append(tout)
        # print(tout.shape)
        
    # print(outs[0])
    return cv2.merge([outs[0],outs[1],outs[2]])


    # print(D.shape)




    


def test_computeTextureWeights(fin=np.ones((5,4,3)),sigma=0,sharpness=0):
    finshape = fin.shape
    vareps_s = sharpness
    vareps = 0.001  
    if len(finshape)==2:
        fx = np.diff(fin)
        col = np.zeros((finshape[0],1))
        fx = np.concatenate((fx,col),axis=1)
        fy = np.diff(fin,axis=0)
        row = np.zeros((1,finshape[1]))
        fy = np.concatenate((fy,row),axis=0)

 
        fx = matCopy(fx)
        fy = matCopy(fy)
    else:
        # b,g,r = cv2.split(fin) 
        fx,fy = mat3tomat3(fin,finshape)


    wto_sum = matSum(np.sqrt(np.square(fx)+np.square(fy)))/3
    print(wto_sum.shape)


def normalize(img):
    datas = cv2.split(img)
    s = []
    for data in datas:
        # m = np.mean(data)
        # mx = np.max(data)
        # mn = np.min(data)
        # ss = 255*(data-np.min(data))/(np.max(data)-np.min(data))
        # s.append(ss)
        s.append(data*255)
    
    return cv2.merge([s[0],s[1],s[2]])



if __name__ == "__main__": 
    img = cv2.imread("/cpfs01/user/puyuandong/glv/Real-ESRGAN/datasets/BSRGAN_degradation/urban100_1/hq/0.png")
    height, width = img.shape[:2]
    # img = cv2.resize(img, (int(width/8), int(height/8)), interpolation=cv2.INTER_CUBIC)

    s = tsmooth(img,maxIter=2)
    s = normalize(s)

    s = np.array(s,dtype=np.uint8)
    cv2.imwrite("example/rtv.png",s)
    