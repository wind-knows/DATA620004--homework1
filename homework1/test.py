# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:49:44 2023

@author: HONOR
"""

import model as my

if __name__ == '__main__':
    # 读取模型文件的路径
    model_path = 'models\\model_v2.npy'
    hidden = 256
    # 读取minist数据集
    train_data, train_label = my.loadMinist(onehot_needed=False)
    test_data, test_label = my.loadMinist(kind='test', onehot_needed=False)
    # 读取模型
    mynet = my.TwoLayerNet(784, hidden, 10, std=1e-4)
    mynet.loadmodel(model_path)
    # 输出模型信息
    mynet.inquire()

    # 计算训练集和测试集上的准确率
    train_result = mynet.predict(train_data)
    train_accuracy = my.test(train_result, train_label) / train_result.shape[0]

    test_result = mynet.predict(test_data)
    test_accuracy = my.test(test_result, test_label) / test_result.shape[0]

    print('训练集上的accuracy：', train_accuracy)
    print('测试集上的accuracy：', test_accuracy)
