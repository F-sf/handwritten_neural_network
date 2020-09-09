# encoding=utf-8
"""
此代码未使用任何框架，实现了神经网络算法的基本原理，用于方便理解
仅依靠CPU进行运算
"""
import mnist_loader
# training_data:( array([0],[0.32551],..784个像素值作为输入x..,[0]),
#                 array([0],[0],..10个输出y表示0-9..) )
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

#import network
import my_network
# 创建一个三层神经网络，一个有784个神经元的输入层，一个有30个神经元的隐藏层，和一个有10个神经元的输出层
#my_net = network.Network([784, 30, 10])
my_net = my_network.MyNetwork([784, 30, 10])

import time
start_time = time.time()
# 使用随机梯度下降算法，迭代次数为5，样本分批10个一批，学习速率3.0
my_net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
end_time = time.time()

# 手写数字识别测试
from cv2 import cv2
import numpy as np
img = cv2.imread('./my_handwrite.png', cv2.IMREAD_GRAYSCALE)
my_data = ((255 - img)/255.0*1.0).reshape([-1,1])
print("手写数字识别测试结果:")
print(np.argmax(my_net.feedforward(my_data)))
print("使用全手撸神经网络，SGD算法，MSE均方差代价函数，同样的网络结构，同样的样本集，同样的迭代次数和mini_batch分批，同样的学习速率")
print("Time Cost:{}s".format(end_time-start_time))