


import sympy
import random as r
# import opt
from numpy import *
from numpy.linalg import inv
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from math import sqrt
from numpy.core.umath import sign
import pandas as pd
from UCIDataSet import UCIDataSet
# Fisher函数
def fisher(x1,x2,test_data,norm='l2'):
    # x1,x2,test_data分别为两类训练样本及待测数据集，其中行为样本数，列为特征数
    # 计算x1,x2样本均值向量 m1,m2
    m1 = mean(x1, axis=0)
    m2 = mean(x2, axis=0)
    sb = m1 - m2

    if norm=='l1':
        s1 = zeros(x1.shape[1])
        for i in x1:
            s1 += i - m1
        s2 = zeros(x2.shape[1])
        for i in x2:
            s2 += i - m2
        sw=s1+s2

        w0 = (ones(x1.shape[1])).reshape(x1.shape[1], 1)
        flag=1
        tol=1e-6
        w = w0
        w_pre = w0
        maxiter=1000
        iter=0
        while flag:
            iter=iter+1
            a=linalg.norm(dot(w_pre,sw.reshape(1,sw.shape[0])),ord=1)/linalg.norm(dot(w_pre,sb.reshape(1,sb.shape[0])),ord=1)
            d1=zeros((sw.shape[0], sw.shape[0]))
            for i in range(x1.shape[1]):
                d1[i][i]=1/linalg.norm(dot(w_pre,sw[i]),ord=1)
            d2=zeros((sb.shape[0], sb.shape[0]))
            for i in range(x1.shape[1]):
                d2[i][i]=1/linalg.norm(dot(w_pre,sb[i]),ord=1)
            # print(d1)
            # flag=0
            ss1=dot(dot(sw.reshape(1,sw.shape[0]),d1),sw.reshape(sw.shape[0],1))
            ss2=dot(dot(sb.reshape(1,sb.shape[0]),d2),sb.reshape(sb.shape[0],1))
            
            w=a*ss2*w_pre/ss1

            # s1=sw.reshape(1,sw.shape[0])*d1*sw.reshape(sw.shape[0],1)
            # s2=sb.reshape(1,sb.shape[0])*d2*sb.reshape(sb.shape[0],1)
            # u, s, v = linalg.svd(s1)
            # s1_inv = dot(dot(v.T, linalg.inv(diag(s))), u.T)
            # w=a*dot(dot(s1_inv,s2),w_pre)
            # w=a*s1_inv*s2*w_pre
            
            if linalg.norm(w-w_pre,ord=2)/linalg.norm(w,ord=2)<=tol or iter>=maxiter:
                print(iter)
                flag=0

            w_pre = w
        print(w)
    else:
        # 计算x1,x2样本类内离散度矩阵s1,s2
        s1 = zeros((x1.shape[1], x1.shape[1]))
        for i in x1:
            t = i - m1
            s1 += t*t.reshape(x1.shape[1], 1)
        s2 = zeros((x2.shape[1], x2.shape[1]))
        for i in x2:
            t = i - m2
            s2 += t*t.reshape(x2.shape[1], 1)
        # 计算总类内离散度矩阵sw
        sw=s1+s2
        # 求投影方向向量 w (维度和样本的维度相同)
        # 即 w=inv(sw)(m1-m2) [公式]
        l2alpha = 0 #L2 正则化超参数
        u, s, v = linalg.svd(sw)
        s_w_inv = dot(dot(v.T, linalg.inv(diag(s))), u.T)
        ss = identity(s_w_inv.shape[0])
        u1, s1, v1 = linalg.svd((ss+l2alpha*s_w_inv))
        s_w_inv1 = dot(dot(v1.T, linalg.inv(diag(s1))), u1.T)

        w1 = dot(s_w_inv1,s_w_inv)
        w = dot(w1, m1 - m2)
        # w = dot(s_w_inv, m1 - m2)
        y1 = dot(w.T, m1)
        y2 = dot(w.T, m2)
        y=[]
        for i in range(test_data.shape[0]):
            pos = dot(w.T, test_data[i])  # 新样本进来判断
            if abs(pos - y1) < abs(pos - y2):
                y.append(1)
            else:
                y.append(0)

        return y


if __name__ == '__main__':
    
    datatype=5 #数据集类型
    isHill_Valley=False
    pathdir='D:\VSCode\SVMtest\mydataset'
    if datatype==1:
        # 制作数据集
        cancer=load_breast_cancer() #用于二分类任务的乳腺癌数据集
        data = cancer.data 
        label = cancer.target
    elif datatype==2:
        # 构造列标签名字
        column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
        # 读取数据  (如果不指定标签名，会默认把第一行数据当成标签名)
        loaddata = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)

        # 缺失值进行处理  (原始数据中的?表示缺失值)
        loaddata = loaddata.replace(to_replace='?', value=nan)
        loaddata = loaddata.dropna()  # 删除有缺少值的行
        data = array(loaddata[column_names[1:10]]) # 第一列是id不需要，最后一列是目标值
        label = array(loaddata[column_names[10]])
        for i in range(label.shape[0]):
            if label[i]==2:
                label[i]=0
            else:
                label[i]=1
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
        X1 = array(data1[column_names[0:11]]) # 最后一列是目标值
        y1 = array(data1[column_names[11]])
        X2 = array(data2[column_names[0:11]]) # 最后一列是目标值
        y2 = array(data2[column_names[11]])
        sampleind1=r.sample(list(arange(0,X1.shape[0])), 250)
        sampleind2=r.sample(list(arange(0,X2.shape[0])), 750)
        X1 = X1[sampleind1]
        y1 = y1[sampleind1]
        X2 = X2[sampleind2]
        y2 = y2[sampleind2]
        print(X1.shape)
        print(X2.shape)
        X=concatenate((X1,X2))
        y=append(y1,y2)
        for i in range(y.shape[0]):
            if y[i]>5:
                y[i]=1
            else:
                y[i]=0
        data=X
        label=y
    elif datatype==6:
        column_names = ['M_ip','S_dip','E_kip','S_ip','M_DM-SNRc','S_dDM-SNRc','E_kDM-SNRc','S_DM-SNRc','Class']
        data = pd.read_csv("D:\VSCode\SVMtest\mydataset\HTRU_2.csv",names=column_names)
        X = array(data[column_names[0:8]]) # 第一列是id不需要，最后一列是目标值
        y = array(data[column_names[8]])
        dataind=arange(0,X.shape[0])
        dataind=list(dataind)
        sampleind=r.sample(dataind, 2000)
        data = X[sampleind]
        label = y[sampleind]
    if isHill_Valley:
        train_data1,test_data1,train_label1,test_label1=train_test_split(xtrain,ytrain,test_size=0.2,random_state=0,stratify=ytrain)
        train_data2,test_data2,train_label2,test_label2=train_test_split(xtest,ytest,test_size=0.2,random_state=0,stratify=ytest)
        train_data,train_label=train_data1,train_label1
        test_data,test_label=test_data2,test_label2
    elif datatype!=3:
        train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.2,random_state=0,stratify=label)
    
    if datatype==2 or datatype==3:
        train_data=train_data.astype(float)
        test_data=test_data.astype(float)
    # #使用单位方差对数据集进行标准化
    # sc=StandardScaler()
    # train_data=sc.fit_transform(train_data)
    # test_data=sc.fit_transform(test_data)
    
    x1=train_data[where(train_label==1)]
    x2=train_data[where(train_label==0)]
    print(x1)
    predictLabel=fisher(x1,x2,test_data,'l2')
    predict = array(predictLabel==test_label)
    acc = sum(predict==1)/len(predict) #正确率
    print("正确率:",acc)
    SD_sample=std(predict) # 标准差
    SE_sample=SD_sample/sqrt(len(predict)) # 标准误差
    print("标准误差:",SE_sample)