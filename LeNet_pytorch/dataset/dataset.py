import gzip, struct
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import math

class dataset():
    def __init__(self,root=None,bath_size=1,shuffle=False):
        self.path=root
        self.bath_size = bath_size
        self.shuffle = shuffle


    def load_mnist(self, prefix='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(self.path, '%s-labels.idx1-ubyte' % prefix)
        images_path = os.path.join(self.path, '%s-images.idx3-ubyte' % prefix)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def CreateDataLoader(self):
        train_img,train_label= self.load_mnist('train')
        test_img,test_label = self.load_mnist('t10k')
#         reshape from (-1,784) to (28,28)
        train_x = torch.from_numpy(train_img.reshape(-1,1,28,28))
        train_y = torch.from_numpy(train_label.astype(int))

        test_x = torch.from_numpy(test_img.reshape(-1,1,28,28))
        test_y = torch.from_numpy(test_label.astype(int))

        train_dataset = TensorDataset(train_x,train_y)
        test_dataset = TensorDataset(test_x,test_y)

        TrainDataloader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=self.bath_size)
        TestDataloader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=self.bath_size)

        return TrainDataloader,TestDataloader
if __name__ == "__main__":
    path =  'E:\ml\LeNet\data'
    batch_size = 1
    shuffle = True
    dataCreate = dataset(path,batch_size,shuffle)
    train_data,test_data = dataCreate.CreateDataLoader()
    for id,(data,label) in enumerate(train_data):
        print('id:%d'%id,data.size())
        torch.save(data,"data.pt")
        torch.save(label,"label.pt")
        break

