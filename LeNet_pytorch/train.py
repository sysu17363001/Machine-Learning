import matplotlib.pyplot as plt # 画loss图像要用的库
import numpy as np # 画图中用到array 存放数据
import torch  # pytorch的库
from dataset.dataset import dataset # 创建数据集的类
from model.model import LeNet # 实现模型的类
from torch.nn import CrossEntropyLoss # loss函数
from torch.optim import Adam # 优化策略

# 画loss曲线
def draw_loss(loss,title="train"):
    epoch = np.arange(len(loss))  # 以epoch为横轴
    plt.plot(epoch,loss) # plot  画曲线图函数
    plt.xlabel("samples") # x轴标签
    plt.ylabel("loss") #y轴标签
    plt.title(title) # 图表标题
    plt.show() # 图结果显示

def train(root,model,batch_size=64,shuffle=True,loss_func=None,optimizer=None,epoch=100):
    #数据集准备
    mnist = dataset(root,batch_size,shuffle) #  实例化一个类
    train_loader,test_loader = mnist.CreateDataLoader() #调用函数得到数据集
    # 迭代训练
    train_loss=[]#记录train的loss
    test_loss=[]# test的loss
    best_accuracy=0 #最好的accuracy记录
    for i in range(epoch):
        print("epoch:",i)
        loss_list1 = 0
        loss_list2 = 0
        # train
        for id,(train_data,train_label) in enumerate(train_loader):
            optimizer.zero_grad() # 每次开始前先清0，不然会积累上一次的
            train_data=train_data.cuda() # 用gpu跑
            train_label=train_label.cuda()
            predicted = model.forward(train_data.float()) # 模型前向传播计算输出的特征向量
            # print(predicted.size())
            loss = loss_func(predicted,train_label.long()) # 计算loss 结果为（batch size,1)的二维张量
            # if id%100==0:
            #     print("id:{},loss:{}".format(id,loss.sum().item()))
            loss_list1 += loss.sum().item() # 将loss变为一个数值
            loss.backward() # 反向梯度计算
            optimizer.step() # 用optimizer对应的优化方法更新所有参数
        loss_list1 = loss_list1/batch_size
        train_loss.append(loss_list1)
        # print("epoch:{},train_loss:{}".format(i, loss_list1))


        correct = 0 #识别正确的个数
        sum =0 # 识别的总的个数
        # test
        for id,(test_data,test_label) in enumerate(test_loader):
            test_data = test_data.cuda()
            test_label = test_label.cuda()

            out = model(test_data.float()).detach()
            loss = loss_func(out,test_label.long())
            loss_list2 += loss.sum().item()/batch_size

            predicted_result = torch.argmax(out,-1) #获取预测的实际对应的标签结果
            # print('predicted result:',predicted_result)
            compare_tensor = (predicted_result == test_label) # 转化为一个对比结果 识别正确返回True，compare_tensor中有多少个true 就分类对了多少个
            # print("is_correct:",compare_tensor)
            correct += torch.sum(compare_tensor.int()) #计算compare_tenso中正确的个数
            sum += compare_tensor.shape[0]
        loss_list2 = loss_list2 / batch_size
        test_loss.append(loss_list2)
        print("epoch:{},train_loss:{},test_loss:{}".format(i,loss_list1, loss_list2))
        # print("correct:",correct)
        # print("sum:",sum)
        print("accuracy:{:.2f}%".format(correct/sum*100))
        print("-------------------------------")
        accuracy = correct / sum
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model, 'log/mnist_accuracy_{:.2f}%_epoch_{}.pkl'.format(correct / sum*100,i))#存一下当前最优模型
    # draw_loss(train_loss,"train") #画loss曲线
    # draw_loss(test_loss,"test")

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