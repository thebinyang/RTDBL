from gabor import *
from dataset import *
import numpy as np
from patch import *
from data_random import *
from broad_learning import *
from skimage.io import imread, imsave,imshow
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from net import *
from edge import *
import torch.nn as nn
import torch.nn.init as init
from scipy.io import loadmat
import time

def onehot(y):

    L=y.shape
    a=L[0]
    onehot_y = np.zeros((a,2),dtype=int)
    u=0
    for i in range(a):

        if y[i]>0:
            onehot_y[i,1]=1
        else:
            onehot_y[i,0]=1
    return np.array(onehot_y)
def faaacc(label1,label2):
    a=accuracy_score(label1,label2)
    b=precision_score(label1,label2)
    c=recall_score(label1,label2)
    d=f1_score(label1,label2)
    return a,b,c,d
def matr(label,pred):
    lenall=len(pred)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(lenall):
        if pred[i]==0 and label[i]==0:
            TN+=1
        elif pred[i]==0 and label[i]==1:
            FN+=1
        elif pred[i]==1 and label[i]==1:
            TP+=1
        elif pred[i]==1 and label[i]==0:
            FP+=1

    PRE = (((TP + FP) * (TP + FN)) + ((FN + TN) * (TN + FP))) / (TP + FP + TN + FN) / (TP + FP + TN + FN)
    OA = (TP + TN) / (TP + FP + TN + FN)
    OE=FP + FN
    KC=(OA - PRE) / (1 - PRE)
    return TN,FN,TP,FP,OE,KC
def predimage(label,pred):
    l=len(label)
    u=np.zeros((l,3))
    for i in range(l):
        if label[i]!=pred[i]:
            if pred[i]==1:
                u[i,2]=255
            else:
                u[i, 0] = 255
        else:
            if pred[i]==1:
                u[i, 0] = 255
                u[i, 1] = 255
                u[i, 2] = 255
    return u
# if __name__ == '__main__':
def main(name):
    print(name)
    s_ = 0.8
    c_ = 0.001
    N1_ = 1100
    N2_ = 1
    N3_ = 7300
    similarity_number_=100
    near_number_=similarity_number_
    rate=0.1
    ceng=3
    modelunet = UNet(ceng=ceng)
    if name=='Italy':
        dir= 'dataset2/Italy'
        data = Data(dir=dir)
        patch_r = 3
        data_pack = pack(data)
        data1, data2, label = data_pack
        feature_data1 = modelunet.cov('sar', torch.FloatTensor(data1).unsqueeze(0).unsqueeze(0))
        feature_data2 = modelunet.cov('opt', torch.FloatTensor(data2).permute(2, 0, 1).unsqueeze(0))
    elif name == 'Img5':
        dir = 'datasets/Img5'
        rate = 0.005
        data = Data3(dir=dir)
        patch_r = 8
        data_pack = pack(data)
        data1, data2, label = data_pack
        feature_data1 = modelunet.cov('sar', torch.FloatTensor(data1).unsqueeze(0).unsqueeze(0))
        feature_data2 = modelunet.cov('opt', torch.FloatTensor(data2).permute(2, 0, 1).unsqueeze(0))
    elif name=='shuguang':
        dir = 'dataset1/shuguang'
        data = Data(dir=dir)
        patch_r = 8
        data_pack = pack(data)
        data1, data2, label = data_pack
        feature_data1 = modelunet.cov('sar', torch.FloatTensor(data1).unsqueeze(0).unsqueeze(0))
        feature_data2 = modelunet.cov('opt', torch.FloatTensor(data2).permute(2, 0, 1).unsqueeze(0))
    # elif name=='Img11':
    #     dir ='datasets/Img11'
    #     data = Data2(dir=dir)
    #     patch_r = 3
    #     data_pack = pack(data)
    #     data1, data2, label = data_pack
    #     feature_data1 = modelunet.cov('opt', torch.FloatTensor(data1).permute(2, 0, 1).unsqueeze(0))
    #     feature_data2 = modelunet.cov('sar', torch.FloatTensor(data2).unsqueeze(0).unsqueeze(0))
    elif name=='4-Img17':

        data=loadmat('datasets/#4-Img17.mat')
        data1=data['image_t1']
        data2=data['image_t2']
        label = data['Ref_gt']
        patch_r = 8
        data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        feature_data1 = modelunet.cov('opt', torch.FloatTensor(data1).permute(2, 0, 1).unsqueeze(0))
        feature_data2 = modelunet.cov('opt', torch.FloatTensor(data2).permute(2, 0, 1).unsqueeze(0))

    elif name=='6-California':
        data=loadmat('datasets/#6-California.mat')
        data1=data['image_t1']
        data2=data['image_t2']
        patch_r = 8
        data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        label =data['Ref_gt']
        feature_data1 = modelunet.cov('opt', torch.FloatTensor(data1).permute(2, 0, 1).unsqueeze(0))
        feature_data2 = modelunet.cov('sarca2', torch.FloatTensor(data2).permute(2, 0, 1).unsqueeze(0))

    H,W=label.shape

    train_dir='/home/zhulian/model/对比算法/CSA/result/'+name+'_train_pos.mat' #由data_random.py文件生成
    train_pos=loadmat(train_dir)
    train_pos=train_pos['train_pos']
    test_dir='/home/zhulian/model/对比算法/CSA/result/'+name+'_test_pos.mat'
    test_pos = loadmat(test_dir)
    test_pos = test_pos['test_pos']
    data1_train, data2_train, label_train = trainingnode(train_pos,feature_data1,feature_data2,label,patch_r)

    label_train_final = onehot(label_train)
    label_patch=label.reshape(H*W)
    ele_num1 = np.sum(label_patch == 0)
    un_train=int(ele_num1*rate)
    undata1_train, undata2_train, unlabel_train = testingnode(train_pos[:un_train,:], feature_data1, feature_data2, label, patch_r)
    pinvOfInput_x1 = pinv(undata1_train, c_)

    pinvOfInput_x2 = pinv(undata2_train, c_)
    Weight_x2 = np.dot(pinvOfInput_x1, undata2_train)
    Weight_x1 = np.dot(pinvOfInput_x2, undata1_train)


    bls=broadlearning(s=s_,c=c_,N1=N1_,N2=N2_,N3=N3_,similarity_number=similarity_number_,near_number=near_number_,all_pos=test_pos,x1_feature=feature_data1,x2_feature=feature_data2,label=label,patch_r=patch_r,Weight_x2=Weight_x2,Weight_x1=Weight_x1,ceng=ceng)

    bls.featurenode(data1_train,data2_train,label_train_final)
    testnumber=90000
    u=(H*W)//testnumber
    predall=[0]*(H*W)
    for t in range(u):
        data1_test, data2_test, label_test = testingnode(test_pos[t*testnumber:(t+1)*testnumber,:], feature_data1, feature_data2, label, patch_r)
        # edge1_test, edge2_test, _ = testingnode(test_pos[t*testnumber:(t+1)*testnumber,:], edge1, edge2, label, pp)
        label_test_final = onehot(label_test)
        pred=bls.test(data1_test,data2_test,label_test_final)
        predall[t*testnumber:(t+1)*testnumber]=pred
    data1_test, data2_test, label_test = testingnode(test_pos[u * testnumber:, :], feature_data1,
                                                     feature_data2, label, patch_r)

    label_test_final = onehot(label_test)
    pred=bls.test(data1_test,data2_test,label_test_final)
    predall[u * testnumber:] = pred
    pred_bls = np.array(predall).reshape(H, W)
    a = np.where((label.reshape(H, W)) > 0, 1, 0)
    b = np.where((pred_bls.reshape(H, W)) > 0, 1, 0)
    OA,pre,re,f1=faaacc(a.reshape(H * W), b.reshape(H * W))
    TN,FN,TP,FP,OE,KC=matr(a.reshape(H * W), b.reshape(H * W))
    image=predimage(a.reshape(H * W), b.reshape(H * W))
    savename=name+'_pred_new.png'

    imsave(savename, image.reshape(H, W,3))
    print(OE)
    print(pre)
    print(re)
    print(f1)
    print(OA)
    print(KC)
    print(TP)
    print(FP)
    print(TN)
    print(FN)

if __name__ == '__main__':
    main('Img5')
    # main('Italy')
    # main('shuguang')
    # main('4-Img17')





