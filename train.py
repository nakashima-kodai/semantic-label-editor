import os, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from utils.visualizer import show_loaded_image
from utils.utils import split_segmap, replace_segmap, merge_class, decode_segmap
np.set_printoptions(threshold=np.inf)


opt = TrainOptions().parse()

### set dataloader ###
print('### prepare DataLoader ###')
data_loader = CreateDataLoader(opt)
train_loader = data_loader.load_data()
dataset_size = len(data_loader)
numof_iteration = len(train_loader)
print('training images : {}'.format(dataset_size))
print('numof_iteration : {}'.format(numof_iteration))


for iter, data in enumerate(train_loader):
    if iter > 0:
        exit()

    batch_label = data['label']
    label1 = batch_label[2]
    label2 = batch_label[0]

    split_label1 = split_segmap(label1)
    split_label2 = split_segmap(label2)

    replaced = replace_segmap(split_label1, split_label2, ch=14)
    # split_label1[:,:,14] = split_label2[:,:,0] # delete
    merged = merge_class(replaced)
    new_label = decode_segmap(merged)

    plt.subplot(1, 2, 1)
    plt.imshow(label1.numpy().transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(new_label)
    plt.show()
