import random as r
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt
from UCIDataSet import UCIDataSet

class Neuralnet():
    def __init__(self, name='nn', datanum=1, layer_structure=[], model=None):
        
        self.name = name
        self.datanum = datanum
        self.layer_number = len(layer_structure) - 1 #神经网络的层数
        self.layer_structure = layer_structure #[输入的特征个数，第1层神经元个数，第2层神经元个数，...，最后一层神经元个数输出层特征个数]
        self.W = []
        self.B = []
        self.total_loss = []
        self.total_accuracy = []
        
        if model==None or model.target == 0:
            # 从头开始初始化网络
            for index in range(self.layer_number):
                self.W.append(np.random.randn(self.layer_structure[index], self.layer_structure[index+1]))
                self.B.append(np.random.randn(1, self.layer_structure[index+1]))
        else:
            # 从训练模型初始化网络
            for index in range(self.layer_number):
                self.W.append(np.array(model.W[index]).reshape(self.layer_structure[index], self.layer_structure[index+1]))
                self.B.append(np.array(model.B[index]).reshape(1, self.layer_structure[index+1]))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_gra(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def forward(self, x):

        # 前向传播 x = [datanum, features]
        self.before_activation = []
        self.activations = [x]
        for index in range(self.layer_number):
            Z = np.dot(self.activations[index], self.W[index]) + self.B[index]
            self.before_activation.append(Z) #每层的输出
            self.activations.append(self.sigmoid(Z)) #每层的激活函数输出
        return self.activations[-1] #输出结果Y

    def lossfunction(self, inputs, target):
        return np.mean(np.sum(-target*np.log(inputs+1e-14) - (1-target)*np.log(1-inputs+1e-14), 1))

    def back_forward(self, targets=None, loss=None, regularization=False):
        # 反向传播
        self.dWs = []
        self.dWsL1 = []
        self.dWsL2 = []
        self.dBs = []
        self.dAs = []
        alpha1=0.01
        alpha2=0.01
        W_reverse = self.W[::-1] #反转
        activations_reverse = self.activations[::-1]
        before_activation_reverse = self.before_activation[::-1]
        # 从最后一层开始往回传播
        for k in range(self.layer_number):
            if(k == 0):
                # 最后一层
                dZ = activations_reverse[k] - targets
                dW = 1/self.datanum*np.dot(activations_reverse[k+1].T, dZ)
                
                dB = 1/self.datanum*np.sum(dZ, axis = 0, keepdims = True)
                dA_before = np.dot(dZ, W_reverse[k].T)
                self.dWs.append(dW)
                self.dBs.append(dB)
                self.dAs.append(dA_before)
                if regularization==True:
                    dWL1=dW+alpha2*np.sign(W_reverse[k])
                    dWL2=dW+alpha1*W_reverse[k]
                    self.dWsL1.append(dWL1)
                    self.dWsL2.append(dWL2)
            else:
                dZ = self.dAs[k-1]*self.sigmoid_gra(before_activation_reverse[k])
                dW = 1/self.datanum*np.dot(activations_reverse[k+1].T,dZ)
                dB = 1/self.datanum*np.sum(dZ, axis = 0, keepdims = True)
                dA_before = np.dot(dZ, W_reverse[k].T)
                self.dWs.append(dW)
                self.dBs.append(dB)
                self.dAs.append(dA_before)
                if regularization==True:
                    dWL1=dW+alpha2*np.sign(W_reverse[k])
                    dWL2=dW+alpha1*W_reverse[k]
                    self.dWsL1.append(dWL1)
                    self.dWsL2.append(dWL2)
        self.dWs = self.dWs[::-1]
        self.dBs = self.dBs[::-1]
        if regularization==True:
            self.dWsL1 = self.dWsL1[::-1]
            self.dWsL2 = self.dWsL2[::-1]
    # 更新每层权重和偏置    
    def steps(self, lr=0.001, lr_decay=False,regultype=0):
        for index in range(len(self.dWs)):
            self.B[index] -= lr*self.dBs[index]
            if regultype==0:
                self.W[index] -= lr*self.dWs[index]
            elif regultype==1:
                self.W[index] -= lr*self.dWsL1[index]
            elif regultype==2:
                self.W[index] -= lr*self.dWsL2[index]

    def train(self, train_datas=None, train_targets=None, train_epoch=1, lr=0.001, lr_decay=False, regularization=False,regultype=0, display=False):
        
        for epoch in range(train_epoch):
            if epoch == int(train_epoch * 0.7) and lr_decay == True:
                lr *= 0.1
            prediction = self.forward(train_datas)
            forward_loss = self.lossfunction(prediction, train_targets)
            accuracy = np.sum((prediction>0.5) == train_targets) / train_targets.shape[0]
            self.total_accuracy.append(accuracy)             
            self.total_loss.append(forward_loss)
            self.back_forward(targets=train_targets, regularization=regularization)
            if regularization==True:
                self.steps(lr=lr, lr_decay=lr_decay,regultype=regultype)
            else:
                self.steps(lr=lr, lr_decay=lr_decay,regultype=0)

    class Model():
        def __init__(self, W, B, target=0):
            self.W = W
            self.B = B
            self.target = target

    def save_model(self):
        model=self.Model(self.W,self.B,1)  
        return model

if __name__ == "__main__":
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
        loaddata = loaddata.replace(to_replace='?', value=np.nan)
        loaddata = loaddata.dropna()  # 删除有缺少值的行
        data = np.array(loaddata[column_names[1:10]]) # 第一列是id不需要，最后一列是目标值
        label = np.array(loaddata[column_names[10]])
        for i in range(label.shape[0]):
            if label[i]==2:
                label[i]=0
            else:
                label[i]=1
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
        sampleind1=r.sample(list(np.arange(0,X1.shape[0])), 250)
        sampleind2=r.sample(list(np.arange(0,X2.shape[0])), 750)
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
        data=X
        label=y
    elif datatype==6:
        column_names = ['M_ip','S_dip','E_kip','S_ip','M_DM-SNRc','S_dDM-SNRc','E_kDM-SNRc','S_DM-SNRc','Class']
        data = pd.read_csv("D:\VSCode\SVMtest\mydataset\HTRU_2.csv",names=column_names)
        X = np.array(data[column_names[0:8]]) # 第一列是id不需要，最后一列是目标值
        y = np.array(data[column_names[8]])
        dataind=np.arange(0,X.shape[0])
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
    
    train_label=train_label.reshape(train_label.shape[0],1)
    test_label=test_label.reshape(test_label.shape[0],1)
    if datatype==2 or datatype==3:
        train_data=train_data.astype(float)
        test_data=test_data.astype(float)
    # 训练次数
    train_epochs = 500
    # 学习率
    lr = 0.01
    regularization = True
    # regultype = 1
    input_features_numbers = train_data.shape[1]
    layer_structure = [input_features_numbers, 16, 8, 1]
    display = True
    net_name = 'nn'
    # 定义我们的神经网络分类器
    net = Neuralnet(name=net_name, datanum=train_data.shape[0],layer_structure=layer_structure)

    pred = (net.forward(test_data))
    # 开始训练
    print("---------开始训练---------")
    for regultype in range(3):
        net.train(train_datas=train_data, train_targets=train_label, train_epoch=train_epochs, lr=lr, regularization=regularization, regultype=regultype,display=display)
        # 保存模型
        model=net.save_model()
        # 测试
        print("---------测试---------")
        # 载入训练好的模型
        net = Neuralnet(name=net_name, datanum=test_data.shape[0],layer_structure=layer_structure, model=model)
        # 网络进行预测
        pred = (net.forward(test_data))
        accu = np.sum((pred>0.5) == test_label) / test_label.shape[0]
        if regultype!=0 and regularization==True:
            print("L%d网络识别的准确率 : "%(regultype), accu)
        else:
            print("网络识别的准确率 : ", accu)
        predict = np.array((pred>0.5)==test_label)
        SD_sample=np.std(predict) # 标准差
        SE_sample=SD_sample/sqrt(len(predict)) # 标准误差
        print("标准误差:",SE_sample)
        