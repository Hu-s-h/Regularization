# coding=utf-8
#########################################
# kNN: k Nearest Neighbors

# 输入:      
#   test_data:待分类数据（测试集数据）
#   train_data:训练集数据
#   train_label:训练集数据的标签
#   k:近邻数

# 输出:     
#   可能性最大的分类标签
#########################################
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import *
import random
import matplotlib.pyplot as plt
import operator
from math import sqrt
import pandas as pd
import numpy as np
from UCIDataSet import UCIDataSet

# KNN分类算法函数定义
def kNNClassify(test_data, train_data, train_label, k,norm='l2'):
    numSamples = train_data.shape[0]   # shape[0]表示行数,即样本数
    # 对测试集计算它与训练集中的每个对象的距离
    diff = tile(test_data, (numSamples, 1)) - train_data  # 按元素求差值
    if norm=='l1':
        squaredDiff = abs(diff)
    else:
        squaredDiff = diff ** 2  # 将差值平方
    squaredDist = sum(squaredDiff, axis = 1)   # 将其按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离

    # 选择距离最近的k个训练对象，作为测试对象的近邻
    sortedDistIndices = argsort(distance)
    classCount = {} #分类用
    # 选择k个最近邻
    for i in range(k):
        voteLabel = train_label[sortedDistIndices[i]]
        # 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 返回出现次数最多的类别标签
    # 根据这k个近邻归属的主要类别，来对测试对象分类
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

if __name__ == '__main__':

    datatype=5 #数据集类型 
    isHill_Valley=False
    pathdir='D:\VSCode\SVMtest\mydataset'
    if datatype==1:
        cancer=load_breast_cancer() #用于二分类任务的乳腺癌数据集
        X = cancer.data 
        y = cancer.target
        # print(y)
    elif datatype==2:
        column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
        # 读取数据  (如果不指定标签名，会默认把第一行数据当成标签名)
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)

        # 缺失值进行处理  (原始数据中的?表示缺失值)
        data = data.replace(to_replace='?', value=np.nan)
        data = data.dropna()  # 删除有缺少值的行
        X = np.array(data[column_names[1:10]]) # 第一列是id不需要，最后一列是目标值
        y = np.array(data[column_names[10]])
    elif datatype==3:
        pathname=['SPECTF.train','SPECTF.test']

        path = pathdir + '\\' + pathname[0]
        raw_dataset = loadtxt(path, delimiter = ',')
        train_data = raw_dataset[:,arange(1,raw_dataset.shape[1])] #特征
        train_label = raw_dataset[:,0] #标签

        path = pathdir + '\\' + pathname[1]
        raw_dataset = loadtxt(path, delimiter = ',')
        test_data = raw_dataset[:,arange(1,raw_dataset.shape[1])] #特征
        test_label = raw_dataset[:,0] #标签
    elif datatype==4:
        isnoise=False
        isHill_Valley=True
        pathname=[]
        if isnoise and isHill_Valley:
            pathname=['Hill_Valley_with_noise_Training.data','Hill_Valley_with_noise_Testing.data']
        elif isHill_Valley:
            pathname=['Hill_Valley_without_noise_Training.data','Hill_Valley_without_noise_Testing.data']
        else:
            pathname=[]
        if isHill_Valley:
            ucidstrain = UCIDataSet(pathdir=pathdir,pathname=pathname[0])
            xtrain,ytrain = ucidstrain.getdata_and_target()
            ucidstest = UCIDataSet(pathdir=pathdir,pathname=pathname[1])
            xtest,ytest = ucidstest.getdata_and_target()
        else:
            ucids = UCIDataSet(pathdir=pathdir,pathname=pathname)
            X,y = ucidstest.getdata_and_target()
    elif datatype==5:
        column_names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
        data1 = pd.read_csv("D:\VSCode\SVMtest\mydataset\winequality-red.csv",names=column_names,delimiter = ';')
        data2 = pd.read_csv("D:\VSCode\SVMtest\mydataset\winequality-white.csv",names=column_names,delimiter = ';')
        X1 = np.array(data1[column_names[0:11]]) # 最后一列是目标值
        y1 = np.array(data1[column_names[11]])
        X2 = np.array(data2[column_names[0:11]]) # 最后一列是目标值
        y2 = np.array(data2[column_names[11]])
        sampleind1=random.sample(list(np.arange(0,X1.shape[0])), 250)
        sampleind2=random.sample(list(np.arange(0,X2.shape[0])), 750)
        X1 = X1[sampleind1]
        y1 = y1[sampleind1]
        X2 = X2[sampleind2]
        y2 = y2[sampleind2]
        print(X1.shape)
        print(X2.shape)
        X=np.concatenate((X1,X2))
        y=np.append(y1,y2)
        for i in range(y.shape[0]):
            if y[i]>5:
                y[i]=1
            else:
                y[i]=0
    elif datatype==6:
      column_names = ['M_ip','S_dip','E_kip','S_ip','M_DM-SNRc','S_dDM-SNRc','E_kDM-SNRc','S_DM-SNRc','Class']
      data = pd.read_csv("D:\VSCode\SVMtest\mydataset\HTRU_2.csv",names=column_names)
      X = np.array(data[column_names[0:8]]) # 最后一列是目标值
      y = np.array(data[column_names[8]])
      dataind=np.arange(0,X.shape[0])
      dataind=list(dataind)
      sampleind=random.sample(dataind, 2000)
      X = X[sampleind]
      y = y[sampleind]

    if isHill_Valley:
        train_data1,test_data1,train_label1,test_label1=train_test_split(xtrain,ytrain,test_size=0.2,random_state=0,stratify=ytrain)
        train_data2,test_data2,train_label2,test_label2=train_test_split(xtest,ytest,test_size=0.2,random_state=0,stratify=ytest)
        train_data,train_label=train_data1,train_label1
        test_data,test_label=test_data2,test_label2
    elif datatype!=3:
        train_data,test_data,train_label,test_label=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)

    #使用单位方差对数据集进行标准化
    sc=StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.fit_transform(test_data)
    # 从训练集中找到和测试数据最接近的k条记录，这里令 k=3
    # k = 3 
    lnorm=['l1','l2']
    for norm in lnorm:
        print('正则化方式：',norm)
        for k in [1,2,3,5,10,15,20]:
        # 调用分类函数对未知数据分类
            print("K=",k)
            predictLabel=[];
            
            for i in range(len(test_data)):
                outputLabel = kNNClassify(test_data[i], train_data, train_label, k,norm)
                predictLabel.append(outputLabel)
            # print("Your input is:", test_data[i], "and classified to class: ", outputLabel)
            predict = array(predictLabel==test_label)

            acc = sum(predict==1)/len(predict) #正确率
            print("正确率:",acc)
            SD_sample=std(predict) # 标准差
            SE_sample=SD_sample/sqrt(len(predict)) # 标准误差
            print("标准误差:",SE_sample)