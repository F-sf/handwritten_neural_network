"""
my_network.py
Brief  : 手撸简单神经网络，激活函数固定为sigmoid，代价函数固定为均方差，训练算法固定为随机梯度下降
Author : FSF
Date   : 2020.9.9
"""
import random
import numpy as np

class MyNetwork(object):
    def __init__(self, size):
        """初始化对象，为权重和偏置赋随机初值 
        size:每层的神经元个数 e.g:[2,3,1]"""
        self.layer_num = len(size)
        # weights和biases的结构为[np矩阵, np矩阵, np矩阵, ...],长度=层数-1,矩阵尺寸为j行k列/j行1列
        # 注意因ws和bs结构不规则,转为np.array也拿不到shape,不如就用list装起来
        self.ws = [np.random.randn(j,k) for j,k in zip(size[1:], size[:-1])]
        self.bs = [np.random.randn(j,1) for j in size[1:]]

    def feedforward(self, a):
        """前向计算，将输入n*1矩阵x送入网络计算输出n*1矩阵y"""
        for i in range(self.layer_num-1):
            a = self.sigmoid(np.dot(self.ws[i], a)+self.bs[i])
        return a

    def SGD(self, train_data, epoch, batch_size, learn_rate, test_data=None):
        """随机梯度下降算法进行训练，按照batch_size进行小批量训练，整体共进行epoch次迭代
        (分批训练并不会减少运行时间,但合理的分批可以极大提高梯度下降的速度)
        train_data  :输入训练数据 e.g:[(x0,y0),(x1,y1),...]xy均为n*1矩阵
        epoch       :整体迭代次数
        batch_size  :将train_data分小批训练时，每批的个数
        learn_rate  :学习速率，即每次梯度下降时乘的系数
        test_data   :与train_data结构相似，区别为y直接是分类结果而非矩阵.若不为None则会在每次整体迭代后进行测试"""
        for epoch in range(epoch):
            random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                self.gradientDescent(batch_data, learn_rate)
            print("epoch{} finish.".format(epoch+1))
            if(test_data): print("test result:{}/{}".format(self.evaluate(test_data), len(test_data)))

    def gradientDescent(self, batch_data, learn_rate):
        """对小批数据执行梯度下降，直接更新self.ws和self.bs"""
        grad_w = [np.zeros(array.shape) for array in self.ws]
        grad_b = [np.zeros(array.shape) for array in self.bs]
        for x,y in batch_data: 
            grad_w_once, grad_b_once = self.backpropagation(x, y)
            for i in range(len(grad_w)):  # 累加整个batch的梯度计算结果
                grad_w[i] += grad_w_once[i]
                grad_b[i] += grad_b_once[i]
        for i in range(len(grad_w)):  # 取均值进行梯度下降
            self.ws[i] -= learn_rate * grad_w[i]/len(batch_data)
            self.bs[i] -= learn_rate * grad_b[i]/len(batch_data)

    def backpropagation(self, x, y):
        """对一个(x,y)数据进行反向传播，返回该次反向传播算出的梯度"""
        grad_w = [np.zeros(array.shape) for array in self.ws]
        grad_b = [np.zeros(array.shape) for array in self.bs]
        # 先进行前向计算保存每一层的z和a
        zs = [np.zeros(array.shape) for array in self.bs]
        a_s = [x]
        a = x  # 第一层的a就是x
        for i in range(self.layer_num-1):
            zs[i] = np.dot(self.ws[i], a) + self.bs[i]
            a = self.sigmoid(zs[i])
            a_s.append(a)
        # 计算最后一层delta
        delta = self.costFuncPrime(a_s[-1], y) * self.sigmoidPrime(zs[-1])
        # 反向传播
        for i in range(self.layer_num-1):
            grad_w[-i-1] = np.dot(a_s[-i-2], delta.transpose()).transpose()
            grad_b[-i-1] = delta
            if i!=self.layer_num-2:  # 最后一轮不需更新第一层的delta
                delta = np.dot(self.ws[-i-1].transpose(), delta) * self.sigmoidPrime(zs[-i-2])
        return grad_w, grad_b

    def evaluate(self, test_data):
        """对test_data中数据依次进行前向计算，返回正确数目"""
        correct_num = 0 
        for x,y in test_data: 
            if np.argmax(self.feedforward(x)) == y: 
                correct_num += 1
        return correct_num

    def costFuncPrime(self, out, target):
        """均方差代价函数的导数"""
        return out - target
    
    def sigmoid(self, x):
        """激活函数sigmoid"""
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self, x):
        """激活函数sigmoid的导数"""
        return self.sigmoid(x)*(1-self.sigmoid(x))