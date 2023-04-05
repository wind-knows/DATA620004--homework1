## 构建两层神经网络分类器

FDU课程 **DATA620004 神经网络和深度学习** 作业1相关代码

胡一航 22210980041


## Dependencies
numpy (1.21.5 √ )


## Code organization
代码框架主要如下：

* `model.py` 实现神经网络的函数及类的定义，包括读取数据、激活函数、前向后向传播、loss函数、SGD更新、训练函数和测试函数等
* `parameter_selection.py` 网格搜索优化超参数（学习率，学习率衰减程度、隐藏层大小，正则化强度）
* `test.py` 测试保存的模型
* `train.py` 自定义训练、保存模型并绘图可视化相应数据


## Run 
### 准备数据与预训练模型
* 下载代码至本地

* 下载[MNIST数据集](https://pan.baidu.com/s/1z7zp9iYkeTdENXcWZmqHQw?pwd=dc9r)（提取码：dc9r）至本地，将其解压到`./minist_data`文件夹中

* 下载[训练好的模型](https://pan.baidu.com/s/1rHlPaHHqIrpPqwWQHbMRxw)（提取码：dpki）至本地，将其解压到`./models`文件夹中

* 注：下载的代码中包含了模型文件，但是数据集不齐（github不让我上传超过25mb的文件），因此只需执行前两步即可

### 测试
运行`test.py`文件可以测试预训练模型在训练集和测试集上的分类精度

可以更改文件中的`model_path`变量以测试不同的模型。

我提供了两个模型，区别在于v1的中间层为1024个神经元而v2为256个。




### 训练
也可以运行`train.py`文件来训练并保存自定义的模型，可更改的变量如下：

* --hidden：中间层神经元个数
* --learning_rate：初始学习率
* --learning_rate_decay：学习率衰减强度(衰减策略为每个epoch后学习率乘上该值)
* --l2：是否启用L2正则化
* --reg_lambda：L2正则化强度
* --model_path：保存模型的路径
* --seed：训练的随机种子
* --batch_size：批大小




