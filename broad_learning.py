from numpy import random
import numpy as np
from sklearn import preprocessing
import math
from scipy import linalg as LA
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# from data_random import *
def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    # mat可以将数组转为矩阵，I是求逆，eye返回的是一个对角线为1其他为0的矩阵
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk
def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1
def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    # maximum逐个对比两个数组中元素，并选择较大的那个
    return z
def two_fans(a):

    all=0
    r=a.shape[1]
    for l in range(r):
        all+=a[0][l]*a[0][l]
    return (math.sqrt(all))/r
def show_accuracy(predictLabel, Label):
    count = 0
    c=0
    p=0
    # print(predictLabel)
    # print(Label)
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j]==1:
            p+=1
        if label_1[j] == predlabel[j]:
            count += 1
            if label_1[j]==1:
                c+=1
    if p==0:
        z=0
        # print(c,p,0)
    else:
        z=c/p
        # print(c,p,c/p)
    return (round(count / len(Label), 5)),z,p,count,len(Label),predlabel
def show_accuracy_test(predictLabel, Label):
    count = 0
    c=0
    p=0
    lab=predictLabel.argmax(axis=1)
    l=predictLabel.shape[0]//2
    pred1=predictLabel[:l,:]
    pred2=predictLabel[l:,:]
    pred=pred1+pred2
    label=Label[:l,:]
    # print(predictLabel)
    # print(Label)
    label_1 = np.zeros(label.shape[0])
    predlabel = []
    label_1 = label.argmax(axis=1)
    predlabel = pred.argmax(axis=1)
    for j in list(range(l)):
        if label_1[j]==1:
            p+=1
        if label_1[j] == predlabel[j]:
            count += 1
            if label_1[j]==1:
                c+=1
    if p==0:
        print(c,p,0)
    else:
        print(c,p,c/p)
    return (round(count /l, 5)),c,p,count,len(Label),predlabel
class broadlearning:
    def __init__(self,s,c,N1,N2,N3,similarity_number,near_number,all_pos,x1_feature,x2_feature,label,patch_r,Weight_x2,Weight_x1,ceng):
        self.s = s
        self.c=c
        self.N3=N3
        self.N1=N1
        self.N2=N2
        self.ceng=ceng
        self.all_pos=all_pos
        self.x1_feature=x1_feature
        self.x2_feature = x2_feature
        self.label = label
        self.patch_r = patch_r
        self.Beta1OfEachWindow_x1 = []
        self.distOfMaxAndMin_x1 = []
        self.minOfEachWindow_x1 = []
        self.ymin = 0
        self.ymax = 1
        self.Beta1OfEachWindow_x2 = []
        self.distOfMaxAndMin_x2 = []
        self.minOfEachWindow_x2 = []
        self.similarity_number=similarity_number
        self.near_number=near_number
        self.weightOfEnhanceLayer = []
        self.Outputweight = []
        self.parameterOfShrink = 0
        self.pinvOfInput=[]
        self.weightOfEnhanceLayer=[]
        self.Weight_x2=Weight_x2
        self.Weight_x1=Weight_x1
        self.x1_si_re=[]
        self.x2_si_re = []
        # self.x1_all=x1_all
        # self.x2_all=x2_all

    def featurenode(self,x1,x2,label):
        # x1_yuan=x1
        # x2_yuan = x2
        x1_re=np.dot(x2, self.Weight_x1)-x1
        x2_re = np.dot(x1, self.Weight_x2) - x2
        t1, t2 = self.similarity_re(x1_re, x2_re)
        # x1=np.hstack([x1,edge1_train])
        # x2=np.hstack([x2,edge2_train])
        FeatureOfInputDataWithBias_x1 = np.hstack([x1, 0.1 * np.ones((x1.shape[0], 1))])
        OutputOfFeatureMappingLayer_x1 = np.zeros([x1.shape[0], self.N2 * (self.near_number + self.N1)])
        FeatureOfInputDataWithBias_x2 = np.hstack([x2, 0.1 * np.ones((x2.shape[0], 1))])
        OutputOfFeatureMappingLayer_x2 = np.zeros([x2.shape[0], self.N2 * (self.near_number + self.N1)])
        # OutputOfFeatureMappingLayer_x1 = np.zeros([x1.shape[0], self.N2 * (self.near_number)])
        # OutputOfFeatureMappingLayer_x2 = np.zeros([x2.shape[0], self.N2 * (self.near_number)])

        # x1
        for i in range(self.N2):
            random.seed(i*20 + 8000)
            weightOfEachWindow_x1 = 2 * random.randn(x1.shape[1] + 1, self.N1) - 1
            FeatureOfEachWindow_x1 = np.dot(FeatureOfInputDataWithBias_x1, weightOfEachWindow_x1)
            scaler1_x1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow_x1)
            FeatureOfEachWindowAfterPreprocess_x1 = scaler1_x1.transform(FeatureOfEachWindow_x1)
            betaOfEachWindow_x1 = sparse_bls(FeatureOfEachWindowAfterPreprocess_x1, FeatureOfInputDataWithBias_x1).T
            self.Beta1OfEachWindow_x1.append(betaOfEachWindow_x1)
            outputOfEachWindow_x1 = np.dot(FeatureOfInputDataWithBias_x1, betaOfEachWindow_x1)
            self.distOfMaxAndMin_x1.append(np.max(outputOfEachWindow_x1, axis=0) - np.min(outputOfEachWindow_x1, axis=0))
            self.minOfEachWindow_x1.append(np.min(outputOfEachWindow_x1, axis=0))

            outputOfEachWindow_x1 = (outputOfEachWindow_x1 - self.minOfEachWindow_x1[i]) / self.distOfMaxAndMin_x1[i]
            #

        #x2
            random.seed(i*20 + 8000)
            weightOfEachWindow_x2 = 2 * random.randn(x2.shape[1] + 1, self.N1) - 1
            FeatureOfEachWindow_x2 = np.dot(FeatureOfInputDataWithBias_x2, weightOfEachWindow_x2)
            scaler1_x2 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow_x2)
            FeatureOfEachWindowAfterPreprocess_x2 = scaler1_x2.transform(FeatureOfEachWindow_x2)
            betaOfEachWindow_x2 = sparse_bls(FeatureOfEachWindowAfterPreprocess_x2, FeatureOfInputDataWithBias_x2).T
            self.Beta1OfEachWindow_x2.append(betaOfEachWindow_x2)
            outputOfEachWindow_x2 = np.dot(FeatureOfInputDataWithBias_x2, betaOfEachWindow_x2)
            self.distOfMaxAndMin_x2.append(
                np.max(outputOfEachWindow_x2, axis=0) - np.min(outputOfEachWindow_x2, axis=0))
            self.minOfEachWindow_x2.append(np.min(outputOfEachWindow_x2, axis=0))
            outputOfEachWindow_x2 = (outputOfEachWindow_x2 - self.minOfEachWindow_x2[i]) / self.distOfMaxAndMin_x2[
                i]
            # t1,t2=self.similarity(outputOfEachWindow_x1,outputOfEachWindow_x2,i)
            OutputOfFeatureMappingLayer_x1[:, (self.N1)* i:(self.N1) * (i + 1)] = outputOfEachWindow_x1
            OutputOfFeatureMappingLayer_x2[:, (self.N1) * i:(self.N1) * (i + 1)] = outputOfEachWindow_x2

            # t1, t2 = self.similarity(outputOfEachWindow_x1, outputOfEachWindow_x2, i)
            # OutputOfFeatureMappingLayer_x1[:,
            # (self.near_number) * i:(self.near_number) * (i + 1)] = t1
            # OutputOfFeatureMappingLayer_x2[:,
            # (self.near_number) * i:(self.near_number) * (i + 1)] = t2
            del outputOfEachWindow_x1
            del FeatureOfEachWindow_x1
            del weightOfEachWindow_x1
            del scaler1_x1
            del FeatureOfEachWindowAfterPreprocess_x1
            del betaOfEachWindow_x1

            del outputOfEachWindow_x2
            del FeatureOfEachWindow_x2
            del weightOfEachWindow_x2
            del scaler1_x2
            del FeatureOfEachWindowAfterPreprocess_x2
            del betaOfEachWindow_x2
        O_x1=OutputOfFeatureMappingLayer_x1
        OutputOfFeatureMappingLayer_x1 = np.hstack([OutputOfFeatureMappingLayer_x1, x1_re])
        # OutputOfFeatureMappingLayer_x1 = np.hstack([OutputOfFeatureMappingLayer_x1])
        InputOfEnhanceLayerWithBias_x1 = np.hstack(
            [OutputOfFeatureMappingLayer_x1, 0.1 * np.ones((OutputOfFeatureMappingLayer_x1.shape[0], 1))])


        if OutputOfFeatureMappingLayer_x1.shape[1] >= self.N3:
            random.seed(67797325)
            # self.weightOfEnhanceLayer = LA.orth(2 * random.randn(self.N2 * (self.near_number +self.N1) + 1+edge1_train.shape[1], self.N3)) - 1
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(OutputOfFeatureMappingLayer_x1.shape[1] + 1, self.N3)) - 1
        else:
            random.seed(67797325)
            # self.weightOfEnhanceLayer = LA.orth(2 * random.randn(self.N2 * (self.near_number +self.N1) + 1+edge1_train.shape[1], self.N3).T - 1).T
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(OutputOfFeatureMappingLayer_x1.shape[1] + 1, self.N3).T - 1).T
        tempOfOutputOfEnhanceLayer_x1 = np.dot(InputOfEnhanceLayerWithBias_x1, self.weightOfEnhanceLayer)
        #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))
        del InputOfEnhanceLayerWithBias_x1
        self.parameterOfShrink = self.s / np.max(tempOfOutputOfEnhanceLayer_x1)

        OutputOfEnhanceLayer_x1 = tansig(tempOfOutputOfEnhanceLayer_x1 * self.parameterOfShrink)
        # output_x1 = np.hstack([OutputOfEnhanceLayer_x1,edge1_train,t1])
        output_x1 = np.hstack([OutputOfEnhanceLayer_x1, t1])
        # output_x1 = np.hstack([OutputOfFeatureMappingLayer_x1, OutputOfEnhanceLayer_x1])
        # del OutputOfFeatureMappingLayer_x1
        del OutputOfEnhanceLayer_x1
        del tempOfOutputOfEnhanceLayer_x1
        O_x2 = OutputOfFeatureMappingLayer_x2
        # OutputOfFeatureMappingLayer_x2 = np.hstack([OutputOfFeatureMappingLayer_x2])
        OutputOfFeatureMappingLayer_x2 = np.hstack([OutputOfFeatureMappingLayer_x2, x2_re])
        InputOfEnhanceLayerWithBias_x2 = np.hstack(
            [OutputOfFeatureMappingLayer_x2, 0.1 * np.ones((OutputOfFeatureMappingLayer_x2.shape[0], 1))])
        tempOfOutputOfEnhanceLayer_x2 = np.dot(InputOfEnhanceLayerWithBias_x2, self.weightOfEnhanceLayer)
        del InputOfEnhanceLayerWithBias_x2
        # parameterOfShrink_x2 = self.s / np.max(tempOfOutputOfEnhanceLayer_x2)
        OutputOfEnhanceLayer_x2 = tansig(tempOfOutputOfEnhanceLayer_x2 * self.parameterOfShrink)
        del tempOfOutputOfEnhanceLayer_x2
        # output_x2 = np.hstack([OutputOfEnhanceLayer_x2,edge2_train,t2])
        output_x2 = np.hstack([OutputOfEnhanceLayer_x2, t2])
        # output_x2 = np.hstack([OutputOfFeatureMappingLayer_x2, OutputOfEnhanceLayer_x2])
        # del OutputOfFeatureMappingLayer_x2
        del OutputOfEnhanceLayer_x2

        output_finalnode = np.hstack([O_x1-O_x2, output_x1, output_x2])
        label_final = label
        self.pinvOfInput = pinv(output_finalnode, self.c)
        self.OutputWeight = np.dot(self.pinvOfInput, label_final)
        OutputOfTrain = np.dot(output_finalnode, self.OutputWeight)
        trainAcc, z, _, _, _, predlabel = show_accuracy(OutputOfTrain, label_final)
        print('Training accurate is', trainAcc * 100, '%')
        # gg=0
        # while gg<4:
        #     OutputOfTrain, label_final,output_finalnode=self.wrongsample(OutputOfTrain, label_final,output_finalnode)
        #     gg+=1
        del OutputOfTrain

    def wrongsample(self,OutputOfTrain, label_final,output_finalnode):
        trainlabel=(np.array(OutputOfTrain)).argmax(axis=1)
        label=label_final.argmax(axis=1)
        indices = np.where(trainlabel != label)[0]
        if len(indices)!=0:
            extranode=output_finalnode[indices[:],:]
            extralabel=label_final[indices[:],:]
            print(extranode.shape[0])
            extranodeT=extranode.T
            # DT = extranode.dot(self.pinvOfInput)
            # CT = extranode - DT.dot(output_finalnode)
            # B = self.pinv(CT) if (CT.T == 0).any() else self.pinvOfInput.dot(DT.T).dot(
            #     np.mat((DT.dot(DT.T) + np.eye(DT.shape[0]))).I)
            #
            # self.OutputWeight = self.OutputWeight + B.dot((extralabel - extranode.dot(self.OutputWeight)))
            # self.pinvOfInput = np.column_stack((self.pinvOfInput - B.dot(DT), B))
            newnode=np.vstack([output_finalnode,extranode])
            newlabel=np.vstack([label_final,extralabel])
            self.pinvOfInput = pinv(newnode, self.c)
            self.OutputWeight = np.dot(self.pinvOfInput, newlabel)
            OutputOfTrainnew = np.dot(newnode, self.OutputWeight)
            trainAcc, z, _, _, _, predlabel = show_accuracy(OutputOfTrainnew, newlabel)
            print('Training accurate is', trainAcc * 100, '%')
            return OutputOfTrainnew,newlabel,newnode





    def test(self,x1,x2,label):

        # x1_yuan = x1
        # x2_yuan = x2
        x1_re = np.dot(x2, self.Weight_x1) - x1
        x2_re = np.dot(x1, self.Weight_x2) - x2
        t1,t2=self.testsimilarity_re(x1_re,x2_re)
        # x1 = x1 + edge1_train
        # x2 = x2 + edge2_train
        # x1 = np.hstack([x1, edge1_train])
        # x2 = np.hstack([x2, edge2_train])
        FeatureOfInputDataWithBias_x1 = np.hstack([x1, 0.1 * np.ones((x1.shape[0], 1))])
        OutputOfFeatureMappingLayer_x1 = np.zeros([x1.shape[0], self.N2 * (self.near_number +self.N1)])
        FeatureOfInputDataWithBias_x2 = np.hstack([x2, 0.1 * np.ones((x2.shape[0], 1))])
        OutputOfFeatureMappingLayer_x2 = np.zeros([x2.shape[0], self.N2 * (self.near_number +self.N1)])
        # OutputOfFeatureMappingLayer_x1 = np.zeros([x1.shape[0], self.N2 * (self.near_number)])
        # OutputOfFeatureMappingLayer_x2 = np.zeros([x2.shape[0], self.N2 * (self.near_number)])
        # x1
        for i in range(self.N2):
            outputOfEachWindow_x1 = np.dot(FeatureOfInputDataWithBias_x1, self.Beta1OfEachWindow_x1[i])
            outputOfEachWindow_x1 = (self.ymax - self.ymin) * (
                    outputOfEachWindow_x1 - self.minOfEachWindow_x1[i]) / self.distOfMaxAndMin_x1[i] - self.ymin
            outputOfEachWindow_x2 = np.dot(FeatureOfInputDataWithBias_x2, self.Beta1OfEachWindow_x2[i])
            outputOfEachWindow_x2 = (self.ymax - self.ymin) * (
                    outputOfEachWindow_x2 - self.minOfEachWindow_x2[i]) / self.distOfMaxAndMin_x2[i] - self.ymin
            # t1, t2 = self.similarity(outputOfEachWindow_x1, outputOfEachWindow_x2, i)
            OutputOfFeatureMappingLayer_x1[:, (self.N1) * i:(self.N1) * (i + 1)] = outputOfEachWindow_x1
            OutputOfFeatureMappingLayer_x2[:, (self.N1) * i:(self.N1) * (i + 1)] = outputOfEachWindow_x2

            # OutputOfFeatureMappingLayer_x1[:,
            # (self.near_number) * i:(self.near_number) * (i + 1)] = t1
            # OutputOfFeatureMappingLayer_x2[:,
            # (self.near_number) * i:(self.near_number) * (i + 1)] = t2
        O_x1 = OutputOfFeatureMappingLayer_x1
        O_x2 = OutputOfFeatureMappingLayer_x2
        OutputOfFeatureMappingLayer_x1 = np.hstack([OutputOfFeatureMappingLayer_x1, x1_re])
        # OutputOfFeatureMappingLayer_x1 = np.hstack([OutputOfFeatureMappingLayer_x1])
        InputOfEnhanceLayerWithBias_x1 = np.hstack(
            [OutputOfFeatureMappingLayer_x1, 0.1 * np.ones((OutputOfFeatureMappingLayer_x1.shape[0], 1))])
        tempOfOutputOfEnhanceLayer_x1 = np.dot(InputOfEnhanceLayerWithBias_x1, self.weightOfEnhanceLayer)
        # parameterOfShrink_x1 = self.s / np.max(tempOfOutputOfEnhanceLayer_x1)
        OutputOfEnhanceLayer_x1 = tansig(tempOfOutputOfEnhanceLayer_x1 * self.parameterOfShrink)
        # output_x1 = np.hstack([OutputOfEnhanceLayer_x1,edge1_train,t1])
        output_x1 = np.hstack([OutputOfEnhanceLayer_x1, t1])
        # output_x1 = np.hstack([OutputOfFeatureMappingLayer_x1, OutputOfEnhanceLayer_x1])

        # OutputOfFeatureMappingLayer_x2 = np.hstack([OutputOfFeatureMappingLayer_x2])
        OutputOfFeatureMappingLayer_x2 = np.hstack([OutputOfFeatureMappingLayer_x2, x2_re])
        InputOfEnhanceLayerWithBias_x2 = np.hstack(
            [OutputOfFeatureMappingLayer_x2, 0.1 * np.ones((OutputOfFeatureMappingLayer_x2.shape[0], 1))])
        tempOfOutputOfEnhanceLayer_x2 = np.dot(InputOfEnhanceLayerWithBias_x2, self.weightOfEnhanceLayer)
        # parameterOfShrink_x2 = self.s / np.max(tempOfOutputOfEnhanceLayer_x2)
        OutputOfEnhanceLayer_x2 = tansig(tempOfOutputOfEnhanceLayer_x2 * self.parameterOfShrink)
        # output_x2 = np.hstack([OutputOfEnhanceLayer_x2,edge2_train,t2])
        output_x2 = np.hstack([OutputOfEnhanceLayer_x2, t2])
        # output_x2 = np.hstack([OutputOfFeatureMappingLayer_x2, OutputOfEnhanceLayer_x2])
        # output_final = np.vstack([math.sqrt(a1) * output_x1, math.sqrt(a2) * output_x2])
        # label_final = np.vstack([math.sqrt(a1) * label, math.sqrt(a2) * label])
        output_final = np.hstack([O_x1-O_x2, output_x1, output_x2])
        label_final = label
        OutputOfTrain = np.dot(output_final, self.OutputWeight)
        testAcc, _, _, _, _, predlabel = show_accuracy(OutputOfTrain, label_final)
        # print(testAcc)
        return predlabel

    def similarity_re(self,x1_re,x2_re):

        number = self.all_pos.shape[0]
        arr = np.arange(number)
        random.seed(120)
        random.shuffle(arr)
        if self.similarity_number != 0:
            pos_si=self.all_pos[arr[:self.similarity_number],:]
            x1_si, x2_si, _ = self.testingnode(pos_si, self.x1_feature, self.x2_feature, self.label, self.patch_r)
            x1_si_re = np.dot(x2_si, self.Weight_x1) - x1_si
            x2_si_re = np.dot(x1_si, self.Weight_x2) - x2_si
            self.x1_si_re=x1_si_re
            self.x2_si_re = x2_si_re

            # Feature_x1_si = np.hstack([x1_si, 0.1 * np.ones((x1_si.shape[0], 1))])
            # out_si_x1= np.dot(Feature_x1_si, self.Beta1OfEachWindow_x1[i_feature])
            # featureout_si_x1 = np.array((self.ymax - self.ymin) * (out_si_x1 - self.minOfEachWindow_x1[i_feature]
            #                                               ) / self.distOfMaxAndMin_x1[i_feature] - self.ymin)
            # Feature_x2_si = np.hstack([x2_si, 0.1 * np.ones((x2_si.shape[0], 1))])
            # out_si_x2= np.dot(Feature_x2_si, self.Beta1OfEachWindow_x2[i_feature])
            # featureout_si_x2 = np.array((self.ymax - self.ymin) * (out_si_x2 - self.minOfEachWindow_x2[i_feature]
            #                                               ) / self.distOfMaxAndMin_x2[i_feature] - self.ymin)
            t1=[]
            t2=[]
            r=(self.patch_r*2+1)*(self.patch_r*2+1)
            cengr=int((x1_si.shape[1])/r)
            g=[]
            for i in range(x1_re.shape[0]):
                f_x1_distance = np.zeros(self.similarity_number*cengr)
                f_x2_distance = np.zeros(self.similarity_number*cengr)
                for j in range(cengr):
                    f1=np.array(np.squeeze(np.array(x1_re[i,r*j:r*(j+1)]))-x1_si_re[:,r*j:r*(j+1)])
                    si_1=np.sum(f1**2, axis=1)
                    f2=np.array(np.squeeze(np.array(x2_re[i,r*j:r*(j+1)]))- x2_si_re[:,r*j:r*(j+1)])
                    si_2=np.sum(f2**2, axis=1)
                    # value_t1, rank_t1 = np.sort(si_1, axis=0), np.argsort(si_1, axis=0)
                    # value_t2, rank_t2 = np.sort(si_2, axis=0), np.argsort(si_2, axis=0)

                    f_x1_distance[j*self.similarity_number:(j+1)*self.similarity_number] = si_1
                    f_x2_distance[j*self.similarity_number:(j+1)*self.similarity_number] = si_2
                    # f_x1_distance[j * self.near_number:(j + 1) * self.near_number] = si_1[0:self.near_number]
                    # f_x2_distance[j * self.near_number:(j + 1) * self.near_number] = si_2[0:self.near_number]

                t1.append(f_x1_distance)
                t2.append(f_x2_distance)
            # t1 = np.hstack([t1, x1_re])
            # t2 = np.hstack([t2, x2_re])
            return t1,t2
        else:
            t1 = []
            # t2 = []
            for i in range(x1_re.shape[0]):
                t1.append([])
            return t1,t1
    def testsimilarity_re(self,x1_re,x2_re):
        # number = self.all_pos.shape[0]
        # arr = np.arange(number)
        # random.seed(120)
        # random.shuffle(arr)
        # pos_si=self.all_pos[arr[:self.similarity_number],:]
        # x1_si, x2_si, _ = self.testingnode(pos_si, self.x1_feature, self.x2_feature, self.label, self.patch_r)
        # x1_si_re = np.dot(x2_si, self.Weight_x1) - x1_si
        # x2_si_re = np.dot(x1_si, self.Weight_x2) - x2_si
        if self.similarity_number!=0:


            t1=[]
            t2=[]
            r = (self.patch_r * 2 + 1) * (self.patch_r * 2 + 1)
            cengr = int((self.x1_si_re.shape[1]) / r)
            g = []
            for i in range(x1_re.shape[0]):
                f_x1_distance = np.zeros(self.similarity_number * cengr)
                f_x2_distance = np.zeros(self.similarity_number * cengr)
                for j in range(cengr):
                    f1 = np.array(np.squeeze(np.array(x1_re[i, r * j:r * (j + 1)])) - self.x1_si_re[:, r * j:r * (j + 1)])
                    si_1 = np.sum(f1 ** 2, axis=1)
                    f2 = np.array(np.squeeze(np.array(x2_re[i, r * j:r * (j + 1)])) - self.x2_si_re[:, r * j:r * (j + 1)])
                    si_2 = np.sum(f2 ** 2, axis=1)
                    # value_t1, rank_t1 = np.sort(si_1, axis=0), np.argsort(si_1, axis=0)
                    # value_t2, rank_t2 = np.sort(si_2, axis=0), np.argsort(si_2, axis=0)

                    f_x1_distance[j * self.similarity_number:(j + 1) * self.similarity_number] = si_1
                    f_x2_distance[j * self.similarity_number:(j + 1) * self.similarity_number] = si_2
                    # f_x1_distance[j * self.near_number:(j + 1) * self.near_number] = si_1[0:self.near_number]
                    # f_x2_distance[j * self.near_number:(j + 1) * self.near_number] = si_2[0:self.near_number]

                t1.append(f_x1_distance)
                t2.append(f_x2_distance)
            # t1 = np.hstack([t1, x1_re])
            # t2 = np.hstack([t2, x2_re])
            return t1, t2
        else:
            t1 = []
            # t2 = []
            for i in range(x1_re.shape[0]):
                t1.append([])
            return t1,t1

    def similarity(self,outputOfEachWindow_x1,outputOfEachWindow_x2,i_feature):
        number = self.all_pos.shape[0]
        arr = np.arange(number)
        random.seed(120+i_feature)
        random.shuffle(arr)
        pos_si=self.all_pos[arr[:self.similarity_number],:]
        x1_si, x2_si, _ = self.testingnode(pos_si, self.x1_feature, self.x2_feature, self.label, self.patch_r)

        # x1_si=np.zeros((self.similarity_number, self.x1_all.shape[1]))
        # x2_si=np.zeros((self.similarity_number, self.x2_all.shape[1]))
        # for i in range(self.similarity_number):
        #     x1_si[i,:]=self.x1_all[arr[i],:]
        #     x2_si[i, :] = self.x2_all[arr[i], :]
        Feature_x1_si = np.hstack([x1_si, 0.1 * np.ones((x1_si.shape[0], 1))])
        out_si_x1= np.dot(Feature_x1_si, self.Beta1OfEachWindow_x1[i_feature])
        featureout_si_x1 = np.array((self.ymax - self.ymin) * (out_si_x1 - self.minOfEachWindow_x1[i_feature]
                                                      ) / self.distOfMaxAndMin_x1[i_feature] - self.ymin)
        Feature_x2_si = np.hstack([x2_si, 0.1 * np.ones((x2_si.shape[0], 1))])
        out_si_x2= np.dot(Feature_x2_si, self.Beta1OfEachWindow_x2[i_feature])
        featureout_si_x2 = np.array((self.ymax - self.ymin) * (out_si_x2 - self.minOfEachWindow_x2[i_feature]
                                                      ) / self.distOfMaxAndMin_x2[i_feature] - self.ymin)
        t1=[]
        t2=[]
        for i in range(outputOfEachWindow_x1.shape[0]):
            f1=(np.squeeze(np.array(outputOfEachWindow_x1[i,:]))-featureout_si_x1)
            si_1=np.sum(f1**2, axis=1)
            f2=(np.squeeze(np.array(outputOfEachWindow_x2[i,:]))- featureout_si_x2)
            si_2=np.sum(f2**2, axis=1)
            value_t1, rank_t1 = np.sort(si_1, axis=0), np.argsort(si_1, axis=0)
            value_t2, rank_t2 = np.sort(si_2, axis=0), np.argsort(si_2, axis=0)
            f_x1_distance = np.zeros(self.near_number)
            f_x2_distance = np.zeros(self.near_number)
            f_x1_distance[0:self.near_number] = si_1[rank_t2[0:self.near_number]]-value_t2[0:self.near_number]
            f_x2_distance[0:self.near_number] = si_2[rank_t1[0:self.near_number]]-value_t1[0:self.near_number]
            # f_x1_distance[0:self.near_number] = si_1[rank_t2[0:self.near_number]]
            # f_x2_distance[0:self.near_number] = si_2[rank_t1[0:self.near_number]]
            t1.append(f_x1_distance)
            t2.append(f_x2_distance)
        t1=np.hstack([t1,outputOfEachWindow_x1-outputOfEachWindow_x2])
        t2 = np.hstack([t2, outputOfEachWindow_x2-outputOfEachWindow_x1])
        return t1,t2
    def testingnode(self,test_pos,x1,x2,label,patch_r):
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






