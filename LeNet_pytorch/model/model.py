from torch import nn

from torchsummary import summary
from dataset.dataset import dataset
class LeNet(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        # 定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1= nn.MaxPool2d(2)
        # 第二个卷积层，6个输入，16个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        # 最后是三个全连接层
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''前向传播函数'''
        # 先卷积，然后调用relu激活函数，再最大值池化操作 (2,2) kernel_size = 2 stride=2
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        # 第二次卷积+池化操作
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # reshape,将多维数据重新变为二维数据，256*400
        x = x.view(-1, 400)
        # print('size', x.size())
        # 第一个全连接
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x




if __name__ == "__main__":
    model = LeNet().cuda()

    path = 'E:\ml\LeNet\data'
    batch_size = 16
    shuffle = True
    dataCreate = dataset(path, batch_size, shuffle)
    train_data, test_data = dataCreate.CreateDataLoader()
    for id, (data, label) in enumerate(train_data):
        predicted = model.forward(data)
        if id == 1:
            break

    summary(model, (1, 28, 28))