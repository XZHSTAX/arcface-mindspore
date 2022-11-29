# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
import numpy as np
import time
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from resnet import *
from ArcModel import Arcface
from MyDataset import get_dataset
from mindspore.dataset import vision,transforms

pic_size=[128,128]
composed = transforms.Compose(
    [
        vision.Normalize(mean=[0.5*255,0.5*255,0.5*255],std=[0.5*255,0.5*255,0.5*255]),
        vision.HWC2CHW()
    ])

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    image = composed(image)
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    
    image = np.concatenate((image, np.fliplr(image)), axis=0)
    image = image.astype(np.float32, copy=False)
    # image -= 127.5
    # image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = Tensor(images)
            output = model(data)
            output = output.asnumpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # feature = output
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, val_dataset, identity_list, compair_list,batch_size=10):
    '''
    Args:
        model : 模型
        val_dataset:   图片绝对路径
        identity_list: 图片相对路径
        compair_list:  lfw的对比文件
        batch_size:    一次推理图片多少
    '''
    s = time.time()
    features, cnt = get_featurs(model, val_dataset,batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}s, average time is {}s/batch'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)             # 图片相对路径 ： feature
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc

ckpt_url = "Arcface_ckpt_new/Arcface_ckpt3/Arcface-5_3560.ckpt"
lfw_test_list = "data/lfw/lfw_test_pair.txt"
lfw_root = "data/lfw/lfw-align-128"
test_batch_size = 10


if __name__ == '__main__':
    net = Arcface(resnet18,512,13938,test=True)  

    param_dict = load_checkpoint(ckpt_url)
    load_param_into_net(net, param_dict)

    identity_list = get_lfw_list(lfw_test_list)
    img_paths = [os.path.join(lfw_root, each) for each in identity_list]

    lfw_test(net, img_paths, identity_list, lfw_test_list,batch_size=64)




