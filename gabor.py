import cv2
import numpy as np
from scipy import ndimage
from skimage.io import imread, imsave,imshow
import torch
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def gabor_filter_bank(ksize, sigma, theta, lambd, gamma):
    filters = []
    for t in theta:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, np.deg2rad(t), lambd, gamma, 0, ktype=cv2.CV_32F)
        filters.append(kernel)
    return filters

def adaptive_convolution(image, filters):
    H,W,Cc=image.shape
    C=Cc-50
    feature_maps = np.zeros((H, W, 4 * C))
    # feature_maps = []
    for c in range(C):
        for i,kernel in enumerate(filters):
            convolved = ndimage.convolve(image[:,:,c], kernel, mode='constant', cval=0.0)
            feature_maps[:,:,c+i]=convolved
    return feature_maps
def nonlinear_structure_tensor(image, ksize=3, alpha=0.04):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(image)

    # 计算图像梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算结构张量
    A = grad_x ** 2
    B = grad_y ** 2
    C = grad_x * grad_y
    # imsave('nst1.png', A)
    # imsave('nst1.png', B)
    # imsave('nst1.png', C)
    # 计算非线性结构张量
    det = A * B - C ** 2
    trace = A + B
    response = det - alpha * trace ** 2

    return response
def compute_tensor(image):
    image=image.squeeze(0).detach().numpy()
    C, H, W = image.shape
    structure_tensor = np.zeros((C,H, W))

    for c in range(C):
        # 计算梯度信息，这里简单使用Sobel算子
        u=nonlinear_structure_tensor(image[c,:,:])
        structure_tensor[c,:, :] = u
    image=image+structure_tensor
    image = torch.tensor(image).unsqueeze(0).float()

    return image

def compute_structure_tensor(image):
    H,W,Cc=image.shape
    C=Cc
    structure_tensor = np.zeros((H, W,3*C))

    for c in range(C):
        # 计算梯度信息，这里简单使用Sobel算子
        grad_x = np.gradient(image[:,:,c], axis=1)
        grad_y = np.gradient(image[:,:,c], axis=0)
        A = grad_x ** 2
        B = grad_y ** 2
        C = grad_x * grad_y
        # imsave('A.png', A)
        # 计算结构性张量的3x3矩阵
        structure_tensor[:,:, c] = A
        structure_tensor[:, :, c+1] = B
        structure_tensor[:, :, c+2] = C

    return structure_tensor
def robert_suanzi(img):
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return img
def edgee(image):
    edges = robert_suanzi(image)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    return eroded_edges
def edgfeature(image):
    H,W,C=image.shape
    structure_tensor = np.zeros((H, W,C))

    for c in range(C):
        # imsave('img.png', image[:,:,c])
        # 计算梯度信息，这里简单使用Sobel算子
        img=edgee(image[:,:,c])
        # 计算结构性张量的3x3矩阵
        structure_tensor[:,:, c] = img
        # imsave('A.png', img)
    return structure_tensor

def robert_suanzi_edge(image):
    image=image.squeeze(0).detach().numpy()
    C, H, W = image.shape
    structure_tensor = np.zeros((C,H, W))

    for c in range(C):
        # 计算梯度信息，这里简单使用Sobel算子
        u=robert_suanzi(image[c,:,:])
        structure_tensor[c,:, :] = u
    image=image+structure_tensor
    image = torch.tensor(image).unsqueeze(0).float()

    return image
