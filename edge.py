import cv2
import numpy as np
from skimage.io import imread, imsave,imshow


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


# # sobel算子的实现
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)  # 无方向算子处理的图像


# Laplace算子
# 常用的Laplace算子模板  [[0,1,0],[1,-4,1],[0,1,0]]   [[1,1,1],[1,-8,1],[1,1,1]]
def Laplace_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_sunnzi))
    return np.uint8(new_image)


