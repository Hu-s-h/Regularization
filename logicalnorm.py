import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from math import sqrt
from sklearn.datasets import load_breast_cancer 
from UCIDataSet import UCIDataSet
import random
 
if __name__=='__main__':
   
   datatype=5 #数据集类型
   isHill_Valley=False
   pathdir='D:\VSCode\SVMtest\mydataset'
   if datatype==1:
      cancer=load_breast_cancer() #用于二分类任务的乳腺癌数据集
      X = cancer.data 
      y = cancer.target
   elif datatype==2:
      # 构造列标签名字
      column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
      # 读取数据  (如果不指定标签名，会默认把第一行数据当成标签名)
      data = pd.read_csv("D:\VSCode\SVMtest\mydataset\\breast-cancer-wisconsin.data", names=column_names)

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

   if isHill_Valley:
      train_data1,test_data1,train_label1,test_label1=train_test_split(xtrain,ytrain,test_size=0.2,random_state=0,stratify=ytrain)
      train_data2,test_data2,train_label2,test_label2=train_test_split(xtest,ytest,test_size=0.2,random_state=0,stratify=ytest)
      train_data,train_label=train_data1,train_label1
      test_data,test_label=test_data2,test_label2
   elif datatype!=3:
      train_data,test_data,train_label,test_label=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
   
   # 进行标准化处理   因为目标结果经过sigmoid函数转换成了[0,1]之间的概率，所以目标值不需要进行标准化。
   std = StandardScaler()
   train_data = std.fit_transform(train_data)
   test_data = std.transform(test_data)
   
   Lnorm = ['l1','l2'] 
   c=[0.1,0.2,0.5,1,2,5,10]
   # 逻辑回归预测
   for norm in Lnorm:
      print('*'*16)
      print('正则化范数：',norm)
      print('*'*16)
      print('-'*16)
      for i in c:
         print('正则化项超参数：',i)
         # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
         lgr = LogisticRegression(C=i,penalty=norm)  
         lgr.fit(train_data, train_label)
   
         # 进行预测
         y_predict = lgr.predict(test_data)
         predict = np.array(y_predict==test_label)
         acc = sum(predict==1)/len(predict) #准确率
         print("准确率:",acc)

         SD_sample=np.std(predict) # 标准差
         SE_sample=SD_sample/sqrt(len(predict)) # 标准误差
         print("标准误差:",SE_sample)
      print('-'*16)
      # print("准确率：", lgr.score(x_test, y_test))  # 0.964912280702
      # print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))
 