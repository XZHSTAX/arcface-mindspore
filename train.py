import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.dataset import vision,transforms

from mindvision.engine.callback import LossMonitor

from MyDataset import get_dataset
from A_softmax import Asoftmax_loss
from resnet import *
from ArcModel import Arcface

lr = 0.0001
decay_rate = 0.1
decay_steps = 100
total_epoch = 1


if __name__ == '__main__':
    image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(64)
    
    net = Arcface(resnet50,512,13938)
    loss_fn = Asoftmax_loss()


    one_epoch_step = train_dataset.get_dataset_size()
    total_step = one_epoch_step*total_epoch

    # 指数下降学习率（floor）
    exponential_decay_lr = nn.ExponentialDecayLR(lr, decay_rate, decay_steps,is_stair=True)
    lr_list = nn.exponential_decay_lr(lr,decay_rate,total_step,decay_steps,1,True)
        
    opt = nn.SGD(net.trainable_params(),learning_rate=exponential_decay_lr)

    # 模型保存callback
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=10)
    ckpoint = ms.ModelCheckpoint(prefix="Arcface", directory="./Arcface_ckpt", config=config_ck)

    # mindisight记录 callback
    specified = {"collect_metric": True, "histogram_regular": "^conv1.*|^conv2.*", "collect_graph": True,
                    "collect_dataset_graph": True}

    summary_collector = ms.SummaryCollector(summary_dir="./summary_dir/summary_01", collect_specified_data=specified,
                                            collect_freq=1, keep_default_action=False, collect_tensor_freq=1)



    model = ms.Model(network=net,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    metrics={"Accuracy": nn.Accuracy()})
    print("-"*20,"traing","-"*20)
    model.train(total_epoch, 
                train_dataset, 
                callbacks=[LossMonitor(lr_list,25),ckpoint],
                dataset_sink_mode=False)

#TODO: 已经把训练过程中要打印的数据模块加入，学习率自动变化加入，模型自动保存加入；下面需要定制mindinsight收集训练过程中的数据

