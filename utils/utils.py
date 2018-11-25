import os
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_cityscapes_labels():
    return np.array([[  0,   0,   0], [128,  64, 128], [244,  35, 232], [ 70,  70,  70],
                     [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170,  30],
                     [220, 220,   0], [107, 142,  35], [152, 251, 152], [ 70, 130, 180],
                     [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
                     [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32]])

def encode_segmap(label):
    label = label.astype(int)
    dst = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    for ii, color_id in enumerate(get_cityscapes_labels()):
        dst[np.where(np.all(label == color_id, axis=-1))[:2]] = ii
    dst = dst.astype(int)
    return dst

def split_class(encoded_label):
    n_class = 20
    dst = np.zeros((encoded_label.shape[0], encoded_label.shape[1], n_class), dtype=np.uint8)
    for iter in range(1, n_class):
        pixels = np.where(encoded_label == iter)
        dst[pixels[0], pixels[1], iter] = 1
    return dst

def split_segmap(label):
    label = (tensor2numpy(label) * 255.0).astype(int)
    encoded_label = encode_segmap(label)
    split_label = split_class(encoded_label)
    return split_label

def tensor2numpy(tensor):
    return tensor.numpy().transpose(1, 2, 0)

def replace_segmap(label1, label2, ch):
    dst = label1.copy()
    dst[:,:,ch] = label2[:,:,ch].copy()
    return dst

def merge_class(splited_label):
    n_class = splited_label.shape[2]
    dst = splited_label[:,:,0].copy()
    for ch in range(1, n_class):
        pixels = np.where(splited_label[:,:,ch] == 1)
        dst[pixels] = ch
    return dst

def decode_segmap(encoded_label):
    n_class = 20
    label_colours = get_cityscapes_labels()
    r = encoded_label.copy()
    g = encoded_label.copy()
    b = encoded_label.copy()
    for ll in range(0, n_class):
        r[encoded_label == ll] = label_colours[ll, 0]
        g[encoded_label == ll] = label_colours[ll, 1]
        b[encoded_label == ll] = label_colours[ll, 2]
    rgb = np.zeros((encoded_label.shape[0], encoded_label.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
