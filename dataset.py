import torch
import glob
import numpy as np
from PIL import Image
# from skimage.io import imread, imsave


dir = 'dataset2/'
# name='shuguang'
# name='Img11'
name='Italy'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU
# H_size = 350
# W_size= 290

class Data(torch.utils.data.Dataset):

    def __init__(self,dir):
        # self.x1 = sorted(glob.glob(dir + name+'-A.ppm'))
        # self.x2 = sorted(glob.glob(dir +name+ '-B.ppm'))
        # self.y = sorted(glob.glob(dir + name+'-C.ppm'))
        self.dir = dir
        self.x1 = sorted(glob.glob(self.dir + '_1.bmp'))
        self.x2 = sorted(glob.glob(self.dir + '_2.bmp'))
        self.y = sorted(glob.glob(self.dir + '_gt.bmp'))

    def __getitem__(self, idx):
        x1 = np.array(Image.open(self.x1[idx]).convert("L"))
        # x1 = np.array(Image.open(self.x1[idx]))
        x2 = np.array(Image.open(self.x2[idx]))
        y = np.array(Image.open(self.y[idx]).convert("L"))

        return x1,x2,y

class Data2(torch.utils.data.Dataset):

    def __init__(self,dir):
        self.dir=dir
        self.x1 = sorted(glob.glob(self.dir+'-A.ppm'))
        self.x2 = sorted(glob.glob(self.dir+ '-B.ppm'))
        self.y = sorted(glob.glob(self.dir+'-C.ppm'))

    def __getitem__(self, idx):
        x1 = np.array(Image.open(self.x1[idx]))
        # x1 = np.array(Image.open(self.x1[idx]))
        x2 = np.array(Image.open(self.x2[idx]).convert("L"))
        y = np.array(Image.open(self.y[idx]).convert("L"))

        return x1,x2,y
def pack(data):

    for y1, y2, l in data:
        y1=y1.astype(np.float32)
        y2 = y2.astype(np.float32)
        # y1[np.abs(y1) <= 0] = np.min(y1[np.abs(y1) > 0])
        # y1 = np.log(y1 + 1)
        # print(np.min(y1))
        y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
        y2 = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))
        l=l/255


    return y1,y2,l
class Data3(torch.utils.data.Dataset):

    def __init__(self,dir):
        self.dir=dir
        self.x1 = sorted(glob.glob(self.dir+'-A.png'))
        self.x2 = sorted(glob.glob(self.dir+ '-B.png'))
        self.y = sorted(glob.glob(self.dir+'-C.png'))

    def __getitem__(self, idx):
        x1 = np.array(Image.open(self.x1[idx]).convert("L"))
        # x1 = np.array(Image.open(self.x1[idx]))
        x2 = np.array(Image.open(self.x2[idx]))
        y = np.array(Image.open(self.y[idx]).convert("L"))

        return x1,x2,y
def pack(data):

    for y1, y2, l in data:
        y1=y1.astype(np.float32)
        y2 = y2.astype(np.float32)
        # y1[np.abs(y1) <= 0] = np.min(y1[np.abs(y1) > 0])
        # y1 = np.log(y1 + 1)
        # print(np.min(y1))
        y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
        y2 = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))
        l=l/255


    return y1,y2,l