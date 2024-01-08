import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
from dataload import ITCDataset, dataset_collate
from net import ITC_net, weights_init
import os

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-------------use %s-----------'%(device))
    '''读取数据集'''

    IE_file = 'car_dataset/trainval/img_emd.txt'
    TE_file = 'car_dataset/trainval/classes_emd.txt'
    labels_file = 'car_dataset/trainval/label.txt'

    batch_size = 256

    dataset = ITCDataset(IE_file, TE_file, labels_file)
    dataloader = DataLoader(dataset, shuffle = False, batch_size = batch_size, num_workers = 8, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate
                            )


    emd_dim = 256
    hiden_dim = 256


    '''定义模型'''
    net = ITC_net(emd_dim, hiden_dim)
    net.to(device)
    net.train()

    '''初始化模型参数'''
    weights_init(net)
    # net = net.to(device)
    # net.train()
    # for layer in net:
    #     if hasattr(layer, 'weight'):
    #         layer.weight.data.normal_(0, 0.01)
    #         layer.bias.data.fill_(0)
    '''定义损失函数'''
    loss = nn.CrossEntropyLoss()

    '''定义优化算法'''
    opt = torch.optim.Adam(net.parameters(),lr=0.005)
    '''训练'''
    num_epochs = 300
    with torch.no_grad():
        text_emd = np.loadtxt(TE_file)
        text_emd = torch.from_numpy(text_emd.astype(np.float32))
        text_emd = text_emd.to(device)
    #print(text_emd.dtype)

    for epoch in range(num_epochs):
        l_all = 0
        for it, batch in enumerate(dataloader):
            X, y1, y2 = batch[0], batch[1], batch[2]
            with torch.no_grad():
                X = X.to(device)
                #print(X.shape)
            y1 = y1.long().to(device)
            y2 = y2.to(device)
            #print(y1)
            #print(label_hat)
            #梯度清零，保证每一步的梯度计算互相不影响          
            l = loss(net(X) @ text_emd.T, y1)
            opt.zero_grad()
            #反向传播，计算梯度
            l.backward()
            #更新参数
            opt.step()
            #print(l.grad)
            l_all += l.item()
        if (epoch+1)%10==0:
            torch.save(net.state_dict(), os.path.join('weights/car/', 'ep%03d-loss%.5f.pth' % (epoch + 1, l_all)))
        print(f'epoch{epoch+1},l_loss:{l_all:f}')


