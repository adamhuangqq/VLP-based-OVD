from cProfile import label
import numpy as np
import pandas as pd
import os

import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
obj_list_voc = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
obj_list = [
    'car','van','truck','suv','bus',
    ]

class ITCDataset(Dataset):
    def __init__(self, IE_file, TE_file, labels_file):
        self.IE_file = IE_file
        self.TE_file = TE_file
        self.label_file = labels_file
        self.IE = np.loadtxt(self.IE_file)
        self.TE = np.loadtxt(self.TE_file)
        with open(self.label_file,'r') as f:
            self.label = f.readlines()
            self.label = [x.split() for x in self.label]
            self.label = [obj_list.index(x[0]) for x in self.label]
        
    def __len__(self):
        # 返回数据的长度
        return len(self.label)

    def __getitem__(self, idx):
        # 返回图像嵌入，标签类别和文本嵌入
        return self.IE[idx], self.label[idx], self.TE[int(self.label[idx])]

def dataset_collate(batch):
    IEs = []
    TEs = []
    labels = []
    for IE, TE, label in batch:
        IEs.append(IE)
        TEs.append(TE)
        labels.append(label)
        
    IEs = torch.from_numpy(np.array(IEs).astype(np.float32))
    TEs = torch.from_numpy(np.array(TEs).astype(np.float32))
    labels = torch.from_numpy(np.array(labels).astype(np.int8))
    return IEs, TEs, labels