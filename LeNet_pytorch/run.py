from train import train
from dataset.dataset import dataset # 创建数据集的类
from model.model import LeNet # 实现模型的类
from torch.nn import CrossEntropyLoss # loss函数
from torch.optim import Adam # 优化策略

if __name__ == "__main__":
    root = './data'
    batch_size = 64
    shuffle = True
    epoch= 200
    # 模型准备
    model = LeNet().cuda()
    # 定义损失函数 优化器
    loss_func = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    epoch = 100 #训练次数
    train(root,model,batch_size,shuffle,loss_func,optimizer,epoch)