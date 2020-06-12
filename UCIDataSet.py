import numpy as np

class UCIDataSet():
    def __init__(self, pathdir, pathname):
        self.pathdir = pathdir
        self.pathname = pathname
    def getdata_and_target(self):
        path = self.pathdir + '\\' + self.pathname
        raw_dataset = np.loadtxt(path, dtype='str', delimiter = ',')
        
        raw_dataset = raw_dataset[1:] #去除第一行
        raw_data = raw_dataset[:,np.arange(0,raw_dataset.shape[1]-1)] #特征
        target = raw_dataset[:,raw_dataset.shape[1]-1] #标签
        data = []
        for i in range(raw_dataset.shape[0]):
            data.append(list(map(float,raw_data[i])))
        data = np.array(data)
        target = np.array(list(map(int,target)))
        return data,target
    