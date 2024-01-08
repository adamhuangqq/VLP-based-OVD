import numpy as np
import os
import torch

#得到类的字典，每个类下面的小字典
def class_dict():
    class_file = 'voc_classes.txt'
    with open(class_file,'r') as f:
        class_list = f.readlines()
        class_list =[x.split('\n')[0] for x in class_list]
    classes_dict = {}
    for cls in class_list:
        classes_dict[cls] = {}
    for key in classes_dict:
        for item in ['tp','tn','fp','fn']:
            classes_dict[key][item] = 0
    return classes_dict

#由txt文件得到标签列表
def file2list(file,mode=0):
    if mode == 0:
        with open(file,'r') as f:
            flist = f.readlines()
        flist = [x.split('\n')[0] for x in flist]
        flist = [x.replace(' ','') for x in flist]
        return flist
    elif mode == 1:
        with open(file,'r') as f:
            flist = f.readlines()
        flist = [x.split()[0] for x in flist]
        return flist

#混淆矩阵
def confusion_matrix(gt_path, detc_path):
    class_file = 'voc_classes.txt'
    with open(class_file,'r') as f:
        class_list = f.readlines()
        class_list =[x.split('\n')[0] for x in class_list]
    length = len(class_list)
    confusion_M = np.zeros([length, length])
    for file in os.listdir(gt_path):
        #print(file)
        gt_list = file2list(gt_path+file,1)
        #print(gt_list)
        detc_list = file2list(detc_path+file,0)
        #print(detc_list)
        for i in range(len(gt_list)):
            gt_index = class_list.index(gt_list[i])
            detc_index = class_list.index(detc_list[i])
            confusion_M[gt_index][detc_index] += 1
    return confusion_M

import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues):
    '''
    cm：        混淆矩阵
    classes：   各个类别
    title：     标题
    '''
    font_dict=dict(fontsize=6,
              color='b',
              family='Times New Roman',
              weight='light',
              style='italic',
              )
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)  
    plt.yticks(tick_marks, classes, rotation=0)
    plt.tick_params(labelsize=6)
    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
        

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 fontdict=font_dict,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    
    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')
    
    plt.tight_layout()
    plt.savefig('res/method_2.png', transparent=True, dpi=1800) 
    
    plt.show()

if __name__=='__main__':
    gt_path = 'gt_path/'
    detc_path = 'detc_path/'

    #confusion_matrix(gt_path, detc_path)
    trans_mat = confusion_matrix(gt_path, detc_path)

    """method 2"""
    if True:
        class_file = 'voc_classes.txt'
        with open(class_file,'r') as f:
            class_list = f.readlines()
            class_list =[x.split('\n')[0] for x in class_list]
        plot_confusion_matrix(trans_mat, class_list)
