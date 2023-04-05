import time
import numpy as np
import matplotlib.pyplot as plt
import model as my

if __name__ == '__main__':

    np.random.seed(123)
    # 超参数
    hidden = 256
    learning_rate = 1e-3
    learning_rate_decay = 0.99  # 学习率衰减策略是每一个epoch后将学习率乘上 learning_rate_decay
    l2 = True
    reg_lambda = 1e-2
    # 保存模型的路径
    model_path = 'models//model_v2'
    # 读取数据
    train_data, train_label = my.loadMinist(onehot_needed=False)
    test_data, test_label = my.loadMinist(kind='test', onehot_needed=False)
    # 我写的 dataloader
    train_dataloader = my.MyDataLoader(train_data, my.oneHot(train_label, 10), batch_size=256, drop_last=True)
    # print(train_dataloader[0])
    # 网络
    mynet = my.TwoLayerNet(784, hidden, 10, std=1e-4)
    mynet.lr = learning_rate
    mynet.l2_lambda = reg_lambda
    mynet.inquire()

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    epoch = 100
    for e in range(epoch):
        # 234个batch 每个batch 256个数据
        print('-' * 50)
        print('epoch:', e + 1, '  start_time:', time.asctime())
        print('')
        start_time = time.time()
        count = 0
        for batch_data, batch_label in train_dataloader:
            loss = mynet.train(batch_data, batch_label, L2reg=l2, lambd=reg_lambda)
            train_loss_list.append(np.mean(loss))
            if count % 30 == 0:  # 每 train 30个 batch 输出一些信息
                print('epoch:', e + 1, '    train counts:', count)
                print('loss:', np.mean(loss))
                print('')
            count += 1
        # 学习率衰减
        mynet.lr = mynet.lr * learning_rate_decay
        # 计算测试集上的 loss
        print('this epoch takes', (time.time() - start_time), 'sec')
        test_result, test_loss = mynet.predict(test_data, my.oneHot(test_label, 10))
        test_loss_list.append(np.mean(test_loss))
        print('test loss:', np.mean(test_loss))
        # 计算训练集上的 accuracy
        train_result = mynet.predict(train_data)
        train_accuracy = my.test(train_result, train_label) / train_result.shape[0]
        train_accuracy_list.append(train_accuracy)
        # 计算测试集上的 accuracy
        count_test = my.test(test_result, test_label)
        test_accuracy = count_test / test_result.shape[0]
        test_accuracy_list.append(test_accuracy)
        print('after this epoch\'s train,the accuracy of test set is: ', test_accuracy)
        print('')

    # 保存模型
    mynet.savemodel(model_path)

    # loss画图
    x = np.linspace(1, epoch, epoch)
    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.plot(x, train_loss_list[0:len(train_loss_list):234], label='train')
    ax1.plot(x, test_loss_list, label='test')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss in Train and Test Dataset')
    ax1.legend()
    plt.savefig('images//{}_loss.jpg'.format(model_path.split('//')[-1]))

    # accuracy画图
    plt.figure(2)
    ax3 = plt.subplot(111)
    ax3.plot(x, train_accuracy_list, label='train')
    ax3.plot(x, test_accuracy_list, label='test')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('accuracy')
    ax3.set_title('Accuracy in Train and Test Dataset')
    ax3.legend()
    plt.savefig('images//{}_accuracy.jpg'.format(model_path.split('//')[-1]))

    # 参数可视化
    W1 = mynet.fc_0.w
    b1 = mynet.fc_0.b
    b1 = np.repeat(b1, 50, axis=1)
    W2 = mynet.fc_1.w
    b2 = mynet.fc_1.b
    b2 = np.repeat(b2, 5, axis=1)
    # w可视化
    plt.figure(3)
    axs = plt.subplot(121)
    axs.imshow(W1)
    axs.set_title('W1')
    plt.xticks([]), plt.yticks([])
    axs = plt.subplot(122)
    axs.imshow(W2)
    axs.set_title('W2')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.savefig('images//{}_parameter_w.jpg'.format(model_path.split('//')[-1]))
    # b可视化
    plt.figure(4)
    axs = plt.subplot(121)
    axs.imshow(b1)
    axs.set_title('b1')
    plt.xticks([]), plt.yticks([])
    axs = plt.subplot(122)
    axs.imshow(b2)
    axs.set_title('b2')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.savefig('images//{}_parameter_b.jpg'.format(model_path.split('//')[-1]))
