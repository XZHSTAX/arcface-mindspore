import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.dataset import vision,transforms

from mindvision.engine.callback import LossMonitor

from MyDataset import get_dataset
from A_softmax import Asoftmax_loss
from resnet import *
from ArcModel import Arcface

if __name__ == '__main__':
    image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(64)

    net = Arcface(resnet50,512,13938)
    loss_fn = Asoftmax_loss()

    # for d,l in train_dataset:
    #     print(d.shape)
    #     output = net(d)
    #     loss = loss_fn(output,l)
    #     print(loss)
        
        
    opt = nn.SGD(net.trainable_params(),learning_rate=0.0001)

    model = ms.Model(network=net,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    metrics={"Accuracy": nn.Accuracy()})

    model.train(1, train_dataset, callbacks=[LossMonitor(0.0001,25)],dataset_sink_mode=False)



