# Arcface-mindspore

此项目为中国科学技术大学图像测量技术课程的课程作业——使用mindspore框架复现论文。论文复现过程中参考了如下两个项目：

[arcface Pytoch](https://github.com/ronghuaiyang/arcface-pytorch)

[arcface-mindspore](https://toscode.gitee.com/mindspore/models/tree/master/official/cv/Arcface)

第一个是个人作者使用pytorch复现的论文，第二个是mindspore官方进行的复现。本想挑一篇没人用mindspore复现过的论文，可惜读论文前没有好好查找，复现debug时才发现了第二个连接，沉没成本过高，只好继续复现。

上面两个复现都非常精彩，尤其mindspore官方的复现，代码非常的工整，优雅。本人的复现就略显粗糙，这个项目的复现目的是尽可能的简单——如果有库，那么就用库，不行就重载，尽量少写自己的代码，尽可能追求一眼就能看出程序与论文对应关系、程序每个模块功能。在这样的思路指导下，项目的介绍展开如下。

# Introdution

本项目实现了arcface的mindspore复现，在CASIA-maxpy-clean数据集上进行训练，在lfw数据集上进行验证。运行环境和平台为Ubuntu 20.04.5 LTS+Python3.9+mindspore1.9+CUDA11.1+3090。

# 复现思路

首先明确，Arcface的主要贡献在于设计了一种新的softmax函数——A-softmax。在pytorch中的交叉熵损失函数会自动对输入进行softmax，在mindspore中也有对应的模型。于是继承`SoftmaxCrossEntropyWithLogits`并重载`construct`，在使用softmax前按照论文方法对输入处理。于是Arcface的核心部分就完成了。

接下来是前面的特征提取网络和线性层。原文使用的是resnet，很明显，按照指导思路，这一块直接使用库函数就好。这一块直接复制[官方复现](https://toscode.gitee.com/mindspore/models/tree/master/official/cv/ResNet)中的`resnet.py`文件，稍作修改即可使用。（后来发现其实完全可以使用`mindvision.classification.models`模型库中的模型，但为了代码的可见性，还是保留原有方案。）

随后，写个类，在resnet后添加一个线性层，把resnet和线性层打包为一个网络。（记得归一化参数）

训练脚本中添加一些超参数、保存和加载路径。测试脚本直接使用了[arcface Pytoch](https://github.com/ronghuaiyang/arcface-pytorch)中的测试脚本，稍作修改，论文就复现完成了。

# 使用说明

文件目录如下：

```
.
├── data
│   ├── CASIA-maxpy-clean
|		├── 0000045
|       └── .... 
│   └── lfw
|       ├── lfw-align-128
|			├── Aaron_Eckhart
|           └── ....
|		└── lfw_test_pair.txt
├── ArcModel.py
├── A_softmax.py
├── Config.py
├── MyDataset.py
├── resnet.py
├── train.py
├── val.py
└── README.md
```

请下载CASIA-maxpy-clean并且解压到文件夹CASIA-maxpy-clean中。下载lfw数据集，并按照上面所示组织文件夹嵌套顺序。

- `train.py`训练脚本，运行即可开始训练
- `val.py` 测试脚本，运行即可测试
- `MyDataset.py`数据集建立脚本
- `A_softmax.py` 损失函数脚本
- `Config.py`纯粹为了欺骗`resnet.py`，不过如果你使用`resnet152`或优化器Thor，记得进去把名字改了
- `resnet.py`官方resnet复现

# Dataset Download

[CASIA-maxpy-clean](https://aistudio.baidu.com/aistudio/datasetdetail/103163)or[CASIA-maxpy-clean](https://onedrive.live.com/?authkey=%21AJjQxHY%2DaKK%2DzPw&cid=1FD95D6F0AF30F33&id=1FD95D6F0AF30F33%2174855&parId=1FD95D6F0AF30F33%2174853&action=locate&sw=bypassConfig)

[lfw](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA )，passward:b2ec 

# Reference

[arcface Pytoch](https://github.com/ronghuaiyang/arcface-pytorch)

[arcface-mindspore](https://toscode.gitee.com/mindspore/models/tree/master/official/cv/Arcface)