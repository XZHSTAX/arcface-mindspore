import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.dataset import vision,transforms

from mindvision.engine.callback import LossMonitor

from MyDataset import get_dataset
from ModelArcface import get_ArcFaceModel


if __name__ == '__main__':
    image_folder_dataset_dir = "data/data_test"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(2)
    # for d,l in train_dataset:
    #     print(l.shape)

    loss = nn.SoftmaxCrossEntropyWithLogits()
    net = get_ArcFaceModel()
    opt = nn.SGD(net.trainable_params(),learning_rate=0.0001)

    model = ms.Model(network=net,
                    loss_fn=loss,
                    optimizer=opt,
                    metrics={"Accuracy": nn.Accuracy()})

    model.train(1, train_dataset, callbacks=[LossMonitor(0.0001)])
#FIXME: 出了点问题，使用自动工具train时，对于Asoftmax前向传播中的label，自动工具好像不会传入这个参数。明天需要细究，可能原因如下：
# 1. 必须要按照官方代码那样，连同loss打包
# 2. 官方复现中我还有没看到的配置，导致了自动工具没有传入
# 3. 如果实在搞不定，就手动训练吧（for），不用自动工具了



