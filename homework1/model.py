import numpy as np
import os
import random


def oneHot(x, n):
    """
    输入：
    x -- np.array,标签向量
    n -- onehot的class数
    
    返回：
    onehot矩阵，一行代表一个label数据
    行数等于len(x) 列数等于n
    """
    return np.eye(n)[x.reshape(-1)]


def reverseOneHot(y):
    """
    反onehot运算，返回列向量
    """
    return np.argmax(y, axis=1).reshape([-1, 1])


# x = np.array([1,2,3,4,2,3,1]).T
# y = oneHot(x,5)
# t = reverseOneHot(y)


def loadMinist(path='./minist_data', kind='train', onehot_needed=True):
    """
    输入：
    path描述数据文件的相对路径
    kind 记录要读取的数据类型：train or test

    return 数据 和 标签(可以选择是否onehot处理)
    """
    if kind == 'test':
        kind = 't10k'

    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))

    with open(images_path) as f_img:
        loaded_img = np.fromfile(file=f_img, dtype=np.uint8)
        images = loaded_img[16:].reshape((-1, 784))

    with open(labels_path) as f_lab:
        loaded_lab = np.fromfile(file=f_lab, dtype=np.uint8)
        labels = loaded_lab[8:].reshape((-1, 1))
    # print(labels[:10])
    if onehot_needed:
        labels = oneHot(labels, 10)

    return images, labels


# train_data, train_label = loadMinist()
# test_data, test_label = loadMinist(kind='test')


def crossEntropyLoss(y_pred, y_true, derivative=False):
    """
    计算交叉熵损失
    输入：
    y_true:真实标签,onehot表示
    y_pred:数据预测出的结果,onehot表示,可以输入多行，每行代表一个数据的预测结果

    如果derivative == True 则返回梯度

    """
    delta = 1e-7  # 防止出现无穷 log(0)
    if not derivative:
        return -np.sum(y_true * np.log(y_pred + delta), axis=1)
    else:
        return -y_true / (y_pred + delta)


# pred = np.array(([0.3,0.7],[0.2,0.8],[0.6,0.4]))
# true = np.array(([0,1],[1,0],[1,0]))
# cel = CrossEntropyLoss(pred,true, derivative=False)
# dcel =  CrossEntropyLoss(pred,true, derivative=True)

'''
下面是sigmoid、softmax、relu三个激活函数

三个函数的架构是类似的，仅输入x则输出对应的激活后的结果
若需要求导的话可以将derivative设置为True,
需要输入x,也可以选择同时输入激活后的结果y，这样会减少求导的运算量
(反正激活函数的结果前面算过了)
'''


#  never use sigmoid and tanh
def sigmoid(x, y=None, derivative=False):
    if derivative:
        if type(y) == type(None):  # 用这种判断是因为 array 不能 if array
            y = sigmoid(x)
        return y * (1 - y)
    return 1 / (1 + np.exp(-x))


def relu(x, y=None, derivative=False):
    if derivative:
        if type(y) != type(None):
            res = y.copy()
            res[y > 0] = 1
            return res
        else:
            y = x.copy()
            y[x > 0] = 1
            y[x < 0] = 0
            return y
    return (abs(x) + x) / 2


def softmax(x, y=None, derivative=False):
    if not derivative:
        row_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - row_max)
        row_sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / row_sum
        return f_x
    else:
        '''
        softmax的求导是比较特殊的，
        假设我们输入是k维的,也即k个class,那么我们的输出也是k维的,
        而k维对k维求导会得到一个k*k的梯度矩阵
        如果我们同时输入n组数据,那么就会有n个矩阵
        这里是计算后输出一个字典dic
        dic[n]得到第n组数据的梯度矩阵
        (我们的输入中每一行是一组数据)
        '''
        if type(y) == type(None):
            y = softmax(x)
        dic = {}
        n, k = x.shape
        # 这里的 n 是数据量   k是class数
        # 记输入为x 输出为y 求导 ∂yi/∂xj 需要分类讨论
        # 如果 i == j ,则是  yi(1-yi)
        # 若果 i != j ,则是  -yiyj
        for num in range(n):
            ans = np.zeros((k, k))
            for i in range(k):
                for j in range(i, k):
                    if i != j:
                        ans[i][j] = - y[num][i] * y[num][j]
                        ans[j][i] = ans[i][j]
                    else:
                        ans[i][j] = y[num][i] * (1 - y[num][j])
            dic[num] = ans
        return dic


# x = np.array(([1,1,1],[4,5,6]))
# y = softmax(x)
# y2 = softmax(x,y,derivative=True)

'''
我写了线性层、激活层、损失层以便后续搭积木使用

下面介绍一些共性的东西：
每个模块都会记录上一次的input/output以供后续反向传播使用
每个模块会用grad记录本次训练反向传播时的的梯度

每个模块都有：
1、forward函数
向前传播，可以用record参数决定是否要让模块记录本次传播
2、backward函数
反向传播，计算模块的梯度并记录
(这里记录的是累计的梯度，也即连乘的结果，因此需要输入后一层的累计梯度)
(模块记录的梯度是loss对该模块input的梯度)
3、update函数
根据现在记录的梯度更新参数，其中activation和loss模块中的update函数不执行任何操作

后续的一些不同的细节会在代码汇中注释
'''


class activation:
    def __init__(self, kind='relu'):
        self.kind = kind  # 激活函数种类,这里可以是relu、sigmoid、softmax
        self.input = None
        self.output = None
        self.grad = None

    def forward(self, x, record=True):
        if record:
            self.input = x
            self.output = eval('{}'.format(self.kind))(x)  # 根据 kind选择激活函数种类
            return self.output
        else:
            return eval('{}'.format(self.kind))(x)

    def __call__(self, x, record=True):
        return self.forward(x, record)

    def backward(self, din):
        # din 是后面传播过来的梯度

        if self.kind in ['relu', 'sigmoid']:
            dself = eval('{}'.format(self.kind))(self.input, self.output, derivative=True)
            ans = dself * din  # 这两种激活函数的反向传播比较简单，矩阵对应位置相乘即刻
            self.grad = ans

        elif self.kind == 'softmax':
            dsoftmax = eval('{}'.format(self.kind))(self.input, self.output, derivative=True)

            h, w = self.input.shape
            ans = np.zeros((h, w))
            # 下面的 i 指的是第 i 条数据
            for i in range(h):
                ans[i] = np.sum((din[i] * dsoftmax[i].T).T, axis=0)
            """ 
            这里是由于softmax求导是个矩阵，
            再反向传播我不知道该怎么写为矩阵形式，所以是一个比较特别的实现
            而softmax作为最后一层再加上交叉熵作为损失函数的话反向传播会有很好的解
            因此下面求导实际没有用到这里
            """

            self.grad = ans
        return self.grad

    def update(self, lr):
        return


""" 
如上文所说，softmax配合交叉熵的话会有较好的解析式
因此这里将softmax和loss函数作为一层，
求导直接跳过对中间结果的导数，会更加简单
"""
class SoftmaxWithCrossEntropyLoss:
    """ Softmax Layer With Cross Entropy Loss """

    def __init__(self):
        self.loss = None
        self.input = None
        self.output = None  # softmax的输出
        self.label = None  # 监督数据
        self.grad = None

    def forward(self, x, label, record=True):
        if record:
            self.input = x
            self.output = softmax(x)
            self.label = label
            self.loss = crossEntropyLoss(self.output, self.label)
            return self.loss
        else:
            return crossEntropyLoss(softmax(x), label)

    def __call__(self, x, label, record=True):
        return self.forward(x, label, record)

    def backward(self, din=1):
        self.grad = self.output - self.label # 非常简单的求导式子 大大加快了运行速度
        return self.grad


class CrossEntropyLoss:
    """ Cross Entropy Loss Layer 实际上没用到，放在这里显得全面一点"""
    def __init__(self):
        self.input = None
        self.output = None  # loss
        self.label = None
        self.grad = None

    def forward(self, x, label, record=True):
        if record:
            self.input = x
            self.label = label
            self.output = crossEntropyLoss(self.input, self.label)
            return self.output
        else:
            return crossEntropyLoss(x, label)

    def __call__(self, x, label, record=True):
        return self.forward(x, label, record)

    def backward(self, din=1):
        self.grad = crossEntropyLoss(self.input, self.label, derivative=True)
        return self.grad


class MyLinear:
    def __init__(self, input_size, output_size, std=1e-4):
        self.w = std * np.random.randn(input_size, output_size)
        self.b = np.zeros((output_size, 1))
        # 我们用 矩阵乘法 模拟全连接层
        # y = w^T @ X ^ T + B
        self.input = None
        self.output = None
        self.grad = None
        self.next_layer_grad = None  # 线性层的更新需要loss对本层output的梯度，因此我们需要额外记录这个

    def forward(self, x, record=True):
        if record:
            self.input = x
            self.output = (np.dot(x, self.w) + self.b.T)
            # 输出依然保证是每行是一个数据
            return self.output
        else:
            return np.dot(x, self.w) + self.b.T

    def __call__(self, x, record=True):
        return self.forward(x, record)

    def backward(self, x):
        self.next_layer_grad = x
        self.grad = np.dot(self.w, x.T).T
        return self.grad

    def update(self, learning_rate):
        n = self.input.shape[0]

        d_b = np.sum(self.next_layer_grad, axis=0).reshape((-1, 1)) / n

        d_w = np.dot(self.input.T, self.next_layer_grad) / n

        self.w = self.w - learning_rate * d_w
        self.b = self.b - learning_rate * d_b

        return

    def __repr__(self):
        # 输出信息方便调试
        print('-' * 50)
        print('')
        print('w:')
        print(self.w)
        print('b:')
        print(self.b)
        print('input:')
        print(self.input)
        print('output:')
        print(self.output)
        print('w.grad:')
        print(self.grad)
        print('next_layer_grad:')
        print(self.next_layer_grad)
        print('')

        return '-' * 50

    """
    两层神经网络：
    输入 X 隐藏 T 输出 Y
    输入为 784 * n 隐藏： k * n 输出 10 * n
    input_size = 784, hidden_size = k, output_size = 10

    input - z1  -  a1  -  z2  -  a2/output
         fc - relu -  fc - softmax
    """


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4, learning_rate=1e-4):
        self.fc_0 = MyLinear(input_size=input_size, output_size=hidden_size, std=std)
        self.ac_0 = activation(kind='relu')
        self.fc_1 = MyLinear(input_size=hidden_size, output_size=output_size, std=std)
        self.sequence = [self.fc_0, self.ac_0, self.fc_1]
        self.lastLayer = SoftmaxWithCrossEntropyLoss()

        self.lr = learning_rate

        self.l2_lambda = None

    def forward(self, x, label=None, record=True):
        # 前向传播
        for layer in self.sequence:
            x = layer(x, record)
        if type(label) != type(None):
            x = self.lastLayer(x, label, record)
        return x

    def backward(self):
        # 反向传播
        din = self.lastLayer.backward(1)
        for layer in self.sequence[::-1]:
            din = layer.backward(din)
        return

    def update(self):
        # 更新
        for layer in self.sequence[::-1]:
            layer.update(self.lr)
        return

    def train(self, data, label, L2reg=False, lambd=1e-2):
        # 交叉熵损失
        loss = self.forward(data, label)

        # 正则项损失
        if L2reg:
            self.l2_lambda = lambd
            m = data.shape[0]  # 样本数
            w0 = self.fc_0.w
            w1 = self.fc_1.w
            L2_regularization_cost = self.l2_lambda * (np.sum(np.square(w0))
                                                       + np.sum(np.square(w1))) / m / 2
            loss += L2_regularization_cost

        # 交叉熵损失反向传播更新参数

        self.backward()
        self.update()
        # 正则惩罚项导致的梯度改变
        if L2reg:
            self.fc_0.w = self.fc_0.w - self.lr * lambd * w0 / m
            self.fc_1.w = self.fc_1.w - self.lr * lambd * w1 / m

        return loss

    def predict(self, data, label=None):
        """

        :param data: 输入数据，根据网络预测结果
        :param label: 数据的label
        :return: 返回预测的结果标签，如果还输入了label，我们还会输出loss
        """
        x = data
        for layer in self.sequence:
            x = layer(x, record=False)
        output = softmax(x)

        result = np.argmax(output, axis=1).reshape((-1, 1))

        if type(label) != type(None):
            loss = crossEntropyLoss(output, label, derivative=False)
            if self.l2_lambda:
                m = data.shape[0]  # 样本数
                w0 = self.fc_0.w
                w1 = self.fc_1.w
                L2_regularization_cost = self.l2_lambda * (np.sum(np.square(w0))
                                                           + np.sum(np.square(w1))) / m / 2
                loss += L2_regularization_cost
            return result, loss
        return result

    def inquire(self):
        # 输出网络的一些信息
        print('-' * 50)
        print('This is a simple two layer neural network')
        print('The whole process is:')
        print('input -- z1 --  a1  -- z2 -- a2/output')
        print('      fc -- relu -- fc -- softmax')
        print('the hidden size is ', self.fc_1.w.shape[0])
        print('the learning rate is ', self.lr)
        if self.l2_lambda is None:
            print('So far we haven\'t train our network or we didn\'t use regularization ')
        else:
            print('we use L2-Regularization, and the lambda is ', self.l2_lambda)
        print('-' * 50)

    def savemodel(self, file):
        # 保存模型
        # 我们只记录参数、学习率和正则化强度
        # file 包含路径和文件名,不要出现 \ , 使用 / 替代
        dic = {'w0': self.fc_0.w, 'b0': self.fc_0.b,
               'w1': self.fc_1.w, 'b1': self.fc_1.b,
               'lr': self.lr,
               'lambd': self.l2_lambda}
        np.save(file, dic)
        return

    def loadmodel(self, file):
        # 读取数据
        b = np.load(file, allow_pickle=True)
        dic = b[()]
        self.fc_0.w = dic['w0']
        self.fc_0.b = dic['b0']
        self.fc_1.w = dic['w1']
        self.fc_1.b = dic['b1']
        self.lr = dic['lr']
        self.l2_lambda = dic['lambd']
        return


# batch数据的生成器

class MyDataLoader:
    def __init__(self, data, label, batch_size, drop_last=False):
        self.data = data
        self.label = label
        self.batch_size = batch_size

        nums = data.shape[0]
        a = [i for i in range(nums)]
        random.shuffle(a)

        self.sampler = a
        self.drop_last = drop_last

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        batch_index = []
        for index in self.sampler:
            batch_index.append(index)
            if len(batch_index) == self.batch_size:
                yield self.data[batch_index], self.label[batch_index]
                batch_index = []
        if len(batch_index) > 0 and not self.drop_last:
            # 如果最后剩余的数据不够一个batch_size,根据参数决定是否 drop out
            yield self.data[batch_index], self.label[batch_index]
        # 每一个epoch后洗牌一次
        random.shuffle(self.sampler)


# 检测预测结果的正确率

def test(result, label):
    if result.shape[0] != label.shape[0]:
        return 'Wrong! Label size != Result size'
    count = 0
    for i in range(result.shape[0]):
        if result[i] == label[i]:
            count += 1
    return count
