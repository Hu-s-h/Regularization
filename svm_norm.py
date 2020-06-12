import sys;
sys.path.append('D:\VSCode\SVMtest\sup');

import random
from math import sqrt
from UCIDataSet import UCIDataSet
import matplotlib.pyplot as plt
import numpy as np
from numpy import std
from sklearn import svm
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *
from sklearn.model_selection import KFold, train_test_split
import pandas as pd

if __name__ == '__main__':
    datatype=4 #数据集类型
    isHill_Valley=False
    pathdir='D:\VSCode\SVMtest\mydataset'
    if datatype==1:
        cancer=load_breast_cancer() #用于二分类任务的乳腺癌数据集
        X = cancer.data 
        y = cancer.target
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
        raw_dataset = np.loadtxt(path, delimiter = ',')
        train_data = raw_dataset[:,np.arange(1,raw_dataset.shape[1])] #特征
        train_label = raw_dataset[:,0] #标签

        path = pathdir + '\\' + pathname[1]
        raw_dataset = np.loadtxt(path, delimiter = ',')
        test_data = raw_dataset[:,np.arange(1,raw_dataset.shape[1])] #特征
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
        X = np.array(data[column_names[0:8]]) # 第一列是id不需要，最后一列是目标值
        y = np.array(data[column_names[8]])
        dataind=np.arange(0,X.shape[0])
        dataind=list(dataind)
        sampleind=random.sample(dataind, 2000)
        X = X[sampleind]
        y = y[sampleind]
    #train_data,test_data,train_label,test_label=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
    # k折划分子集
    resultacc=[]
    resultse=[]
    se_sample=[]
    se_samplenorm=[]
    acc=[]
    acctrain=[]
    acctest=[]
    norm=['l1','l2']
    cc=np.logspace(-2,1) #等比数列

    if datatype==3:
        for p in norm:
            # print('惩罚项为:%s' %p)
            train_scores=[]
            test_scores=[]
            se_score=[]
            for c in cc :
                cls=svm.LinearSVC(penalty=p,C=c,dual=False)
                cls.fit(train_data,train_label)

                predict_label=cls.predict(test_data)
                predict = np.array(predict_label==test_label)
                SD_sample=std(predict) #标准差
                SE_sample=SD_sample/sqrt(len(predict)) ##标准误差
                se_score.append(SE_sample)
                train_scores.append(cls.score(train_data,train_label))
                test_scores.append(cls.score(test_data,test_label))

                # if c==cc[(len(cc)-1)//2]:
                #     resultacc.append(cls.score(test_data,test_label))
                #     resultse.append(SE_sample)
                #     print("算法评分(分类正确率)：%.3f" % cls.score(test_data,test_label))
                #     print("算法标准误差：%.3f" % SE_sample)
                print('罚项系数C:%g' %c)
                # print('个特征权重w：%s,截距b：%s' %(cls.coef_,cls.intercept_))
                print("%s算法评分(分类正确率)：%.3f" % (p,cls.score(test_data,test_label)))
                print("算法标准误差：%.3f" % SE_sample)
                print('\n')
            print(test_scores)
            print(se_score)
    else:
        kf = KFold(n_splits=10) #10折交叉验证
        if isHill_Valley:
            usedata = xtrain
        else:
            usedata = X
        for train, test in kf.split(usedata):
            if isHill_Valley:
                train_data, test_data=xtrain[train], xtest[test]
                train_label, test_label = ytrain[train], ytest[test]
            else:
                train_data, test_data = X[train], X[test]
                train_label, test_label = y[train], y[test]
            # print("k折划分：%s %s" % (train.shape, test.shape))
        
            cls=svm.LinearSVC(loss='hinge',max_iter=1000)
            cls.fit(train_data,train_label)

            predict_label=cls.predict(test_data)
            predict = np.array(predict_label==test_label)
            SD_sample=std(predict) #标准差
            se_sample.append(SD_sample/sqrt(len(predict)))##标准误差
            acc.append(cls.score(test_data,test_label))

            # L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
            # L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合
            # 正则化系数C，允许划分错误的权重（越大，越不允许出错），当C较小时，允许少量样例划分错误
            
            normtrain_scores=[]
            normtest_scores=[]
            normse_score=[]
            for p in norm:
                # print('惩罚项为:%s' %p)
                train_scores=[]
                test_scores=[]
                se_score=[]
                for c in cc :
                    cls=svm.LinearSVC(penalty=p,C=c,dual=False)
                    cls.fit(train_data,train_label)

                    predict_label=cls.predict(test_data)
                    predict = np.array(predict_label==test_label)
                    SD_sample=std(predict) #标准差
                    SE_sample=SD_sample/sqrt(len(predict)) ##标准误差
                    se_score.append(SE_sample)
                    train_scores.append(cls.score(train_data,train_label))
                    test_scores.append(cls.score(test_data,test_label))

                    if c==cc[(len(cc)-1)//4]:
                        resultacc.append(cls.score(test_data,test_label))
                        resultse.append(SE_sample)
                        print(c)
                        print("%s算法评分(分类正确率)：%.3f" % (p,cls.score(test_data,test_label)))
                        print("算法标准误差：%.3f" % SE_sample)
                    # print('罚项系数C:%g' %c)
                    # print('个特征权重w：%s,截距b：%s' %(cls.coef_,cls.intercept_))
                    # print("算法评分(分类正确率)：%.3f" % cls.score(test_data,test_label))
                    # print("算法标准误差：%.3f" % SE_sample)
                    # print('\n')
                normtrain_scores.append(train_scores)
                normtest_scores.append(test_scores)
                normse_score.append(se_score)
            acctrain.append(normtrain_scores)
            acctest.append(normtest_scores)
            se_samplenorm.append(normse_score)
        se = np.mean(se_sample)
        accuracy = np.mean(acc)
        acctrainmean=np.array(acctrain[0])
        # print(np.array(acctrainmean).shape)
        for i in range(len(acctrain)-1):
            acctrainmean+=np.array(acctrain[i+1])
        acctrainmean/=len(acctrain)
        # print(acctrainmean.shape)

        acctestmean=np.array(acctest[0])
        # print(np.array(acctestmean).shape)
        for i in range(len(acctest)-1):
            acctestmean+=np.array(acctest[i+1])
        acctestmean/=len(acctest)
        # print(acctestmean.shape)

        se_samplenormmean=np.array(se_samplenorm[0])
        # print(np.array(se_samplenormmean).shape)
        for i in range(len(se_samplenorm)-1):
            se_samplenormmean+=np.array(se_samplenorm[i+1])
        se_samplenormmean/=len(se_samplenorm)
        # print(se_samplenormmean.shape)
        print('acc:',resultacc)
        print('se:',resultse)
        #绘图表示  
        for i in range(2):
            figue=plt.figure()
            f=figue.add_subplot(1,1,1)
            f.plot(cc,acctrainmean[i],label="L%d Training Accuracy"%(i+1))
            f.plot(cc,acctestmean[i],label="L%d Testing Accuracy"%(i+1))
            f.set_xlabel(r'c')
            f.set_ylabel(r'accuracy')
            f.set_xscale('log')
            f.set_title("L%d SVM"%(i+1))
            f.legend(loc='best')
            plt.show()
        figue=plt.figure()
        f=figue.add_subplot(1,1,1)
        f.plot(cc,acctestmean[0],label="L%d Testing Accuracy"%(1))
        f.plot(cc,acctestmean[1],label="L%d Testing Accuracy"%(2))
        f.set_xlabel(r'c')
        f.set_ylabel(r'accuracy')
        f.set_xscale('log')
        f.set_title("L1 and l2 SVM")
        f.legend(loc='best')
        plt.show()

        figue=plt.figure()
        f=figue.add_subplot(1,1,1)
        f.plot(cc,se_samplenormmean[0],label="L%d SE"%(1))
        f.plot(cc,se_samplenormmean[1],label="L%d SE"%(2))
        f.set_xlabel(r'c')
        f.set_ylabel(r'se')
        f.set_xscale('log')
        f.set_title("L1 and l2 SVM SE")
        f.legend(loc='best')
        plt.show()
