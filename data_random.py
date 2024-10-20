import numpy as np
import random
from scipy.io import savemat
def arr1(length):
    arr = np.arange(length)
    # print(arr)
    random.seed(6383)
    random.shuffle(arr)
    # print(arr)
    return arr
def arr2(length):
    arr = np.arange(length)
    # print(arr)
    random.seed(68831)
    random.shuffle(arr)
    # print(arr)
    return arr
def arr3(length):
    arr = np.arange(length)
    # print(arr)
    random.seed(63288)
    random.shuffle(arr)
    # print(arr)
    return arr

def createTrainingnodes(X1, X2, y,rate,pos):
    ele_num1 = np.sum(y == 0)
    ele_num2 = np.sum(y == 1)
    patch_X1_unchange = np.zeros((ele_num1,  X1.shape[1]))
    patch_X2_unchange = np.zeros((ele_num1,  X2.shape[1]))
    y_unchange = np.zeros(ele_num1)

    patch_X1_change = np.zeros((ele_num2, X1.shape[1]))
    patch_X2_change = np.zeros((ele_num2, X2.shape[1]))
    y_change = np.zeros(ele_num2)
    p=0
    q=0
    unindex=[]
    chindex=[]
    for i in range(X1.shape[0]):
        if y[i]==0:
            patch_X1_unchange[p,:]=X1[i,:]
            patch_X2_unchange[p,:]=X2[i,:]
            y_unchange[p]=y[i]
            p+=1
            unindex.append(i)
        else:
            patch_X1_change[q,:]=X1[i,:]
            patch_X2_change[q,:]=X2[i,:]
            y_change[q]=y[i]
            q+=1
            chindex.append(i)
    # 调用arr函数打乱数组
    arr_1 = arr1(len(patch_X1_unchange))
    arr_2 = arr2(len(patch_X1_change))
    train1=int(len(patch_X1_unchange)*rate)
    train2 = int(len(patch_X1_change) * rate)
    g=arr_1[0:train1]
    f=arr_2[0:train2]
    index=[]
    pos_train=[]
    for p in g:
        index.append(unindex[p])
        pos_train.append(pos[unindex[p]])
    for q in f:
        index.append(chindex[q])
        pos_train.append(pos[chindex[q]])
    # index=np.array(index)
    # index={'train_index':index}
    # savemat('/home/zhulian/model/对比算法/CSA/result/6-California_train_index.mat',index)
    # index = {'test_index':np.arange(ele_num1+ele_num2)}
    # savemat('/home/zhulian/model/对比算法/CSA/result/6-California_test_index.mat',index)
    # index={'train_pos':np.array(pos_train)}
    # savemat('/home/zhulian/model/对比算法/CSA/result/6-California_train_pos.mat', index)
    # index = {'test_pos': np.array(pos)}
    # savemat('/home/zhulian/model/对比算法/CSA/result/6-California_test_pos.mat', index)
    # train={'unchange_train':g,'change_train':f}
    # savemat('shuguang_train.mat', train)

    train_len=train1+train2

    pdata1 = np.zeros((train_len, X1.shape[1]))
    pdata2 = np.zeros((train_len, X2.shape[1]))
    plabels = np.zeros(train_len)

    for i in range(train1):
        pdata1[i, :] = patch_X1_unchange[arr_1[i], :]
        pdata2[i, :] = patch_X2_unchange[arr_1[i], :]
        plabels[i] = y_unchange[arr_1[i]]
    for j in range(train1, train_len):
        pdata1[j, :] = patch_X1_change[arr_2[j - train1], :]
        pdata2[j, :] = patch_X2_change[arr_2[j - train1], :]
        plabels[j] = y_change[arr_2[j - train1]]
    arrz = arr3(len(pdata1))
    finldata1 = np.zeros((train_len, X1.shape[1]))
    finldata2 = np.zeros((train_len, X2.shape[1]))
    finllabels = np.zeros(train_len)
    for u in range(train_len):
        finldata1[u, :] = pdata1[arrz[u], :]
        finldata2[u, :] = pdata2[arrz[u], :]
        finllabels[u] = plabels[arrz[u]]
    return finldata1,finldata2, finllabels
    # return pdata1,pdata2,plabels
def trainingnode(train_pos,x1,x2,label,patch_r):
    train_len=train_pos.shape[0]
    arrz = arr3(train_len)
    H, W,C=x1.shape
    x11=[]
    for c in range(C):
        xx = np.pad(x1[:,:,c], (patch_r, patch_r), "constant", constant_values=0)
        x11.append(xx)
    x22 = []
    for c in range(C):
        xx = np.pad(x2[:, :, c], (patch_r, patch_r), "constant", constant_values=0)
        x22.append(xx)
    patch_size=2*patch_r+1
    p=[]
    x11=np.array(x11)
    x22 = np.array(x22)
    x1_train=[]
    x2_train=[]
    label_train=[]
    for i in range(train_len):
        u1 = (x11[:, train_pos[arrz[i]][0]:train_pos[arrz[i]][0] + patch_size, train_pos[arrz[i]][1]:train_pos[arrz[i]][1] + patch_size]).flatten().tolist()
        x1_train.append(u1)
        u2 = (x22[:, train_pos[arrz[i]][0]:train_pos[arrz[i]][0] + patch_size, train_pos[arrz[i]][1]:train_pos[arrz[i]][1] + patch_size]).flatten().tolist()
        x2_train.append(u2)
        label_train.append(label[train_pos[arrz[i]][0],train_pos[arrz[i]][1]])
    return np.array(x1_train),np.array(x2_train),np.array(label_train)

def testingnode(test_pos,x1,x2,label,patch_r):
    test_len=test_pos.shape[0]
    # arrz = arr3(train_len)
    H, W,C=x1.shape
    x11=[]
    for c in range(C):
        xx = np.pad(x1[:,:,c], (patch_r, patch_r), "constant", constant_values=0)
        x11.append(xx)
    x22 = []
    for c in range(C):
        xx = np.pad(x2[:, :, c], (patch_r, patch_r), "constant", constant_values=0)
        x22.append(xx)
    patch_size=2*patch_r+1
    p=[]
    x11=np.array(x11)
    x22 = np.array(x22)
    x1_test=[]
    x2_test=[]
    label_test=[]
    for i in range(test_len):
        u1 = (x11[:, test_pos[i][0]:test_pos[i][0] + patch_size,
              test_pos[i][1]:test_pos[i][1] + patch_size]).flatten().tolist()
        x1_test.append(u1)
        u2 = (x22[:, test_pos[i][0]:test_pos[i][0] + patch_size,
              test_pos[i][1]:test_pos[i][1] + patch_size]).flatten().tolist()
        x2_test.append(u2)
        label_test.append(label[test_pos[i][0],test_pos[i][1]])
    return np.array(x1_test),np.array(x2_test),np.array(label_test)


# def trainingpatch