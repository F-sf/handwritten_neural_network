# encoding:utf-8
""" pytorch下同结构神经网络测试 """
import torch
import numpy as np
import mnist_loader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import copy

# training_data:( array([0],[0.32551],..784个像素值作为输入x..,[0]),
#                 array([0],[0],..10个输出y表示0-9..) )
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_cpy = copy.deepcopy(training_data)
print(1)
train_x_np = [data[0] for data in training_data_cpy]
train_y_np = [data[1] for data in training_data]
print(2)
train_x = torch.tensor(train_x_np).type(torch.FloatTensor)
train_y = torch.tensor(train_y_np).type(torch.FloatTensor)
print(3)
train_x = train_x.reshape(50000, 784)
train_y = train_y.reshape(50000, 10)
print(4)

# 数据集小批训练，10个为一批，每次随机洗牌
torch_dataset = Data.TensorDataset(train_x, train_y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=10, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.output = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, a_input):
        a_hidden = torch.sigmoid(self.hidden(a_input))      # activation function for hidden layer
        a_output = torch.sigmoid(self.output(a_hidden))
        return a_output

# function 1
net = Net(n_feature=784, n_hidden=30, n_output=10)     # define the network
# function 2(quicker)
_net = torch.nn.Sequential(
    torch.nn.Linear(784, 30),
    torch.nn.Sigmoid(),
    torch.nn.Linear(30, 10),
    torch.nn.Sigmoid(),
)

optimizer = torch.optim.SGD(net.parameters(), lr=3.0)
loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted

import time
start_time = time.time()
for epoch in range(5):  # 5个epoch
    print("迭代次数：", epoch)
    for step, (b_x, b_y) in enumerate(loader):  # 小批次训练
        out = net(b_x)                 # input x and predict based on x
        loss = loss_func(out, b_y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
end_time = time.time()

# 手写数字识别测试
from cv2 import cv2
img = cv2.imread('./my_handwrite.png', cv2.IMREAD_GRAYSCALE)
my_data = torch.tensor(((255 - img)/255.0*1.0).reshape(-1)).type(torch.FloatTensor)
print("手写数字识别测试结果:")
print(torch.max(net.forward(my_data), 0)[1])
print("使用pytorch，SGD算法，MSE均方差代价函数，同样的网络结构，同样的样本集，同样的迭代次数和mini_batch分批，同样的学习速率")
print("Time Cost:{}s".format(end_time-start_time))