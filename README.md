[TOC]



# BLIP-main

## test_*.py文件

这些文件都是自己基于BLIP模型针对下游任务搭建的微调模型，在本文章中，主要使用的是ITC微调模型，主要用到的几个文件的作用如下：

​	test_ITC_batch.py：对建议框进行ITC匹配标签，zero-shot，分为加prompt和不加prompt

​	test_ITC_batch_ft.py：对建议框进行ITC匹配标签，加入了微调网络，加了prompt，微调网络在ITC_finetune文件夹

​	test_emd_batch.py：把嵌入层的数据拿出来，作为微调网络的输入

​	data/dataloader_emd.py和data/dataloader_file.py都是上述文件用到的输入

# faster-rcnn-pytorch-master

这个文件夹里面是普通的faster-rcnn的代码，里面有修修改的地方是roipooling换成了roialign。

# ITC_finetune

这个文件夹是BLIP的微调网络，采用BLIP的输出图像嵌入作为输入，类别文本嵌入作为输出，采用对比学习的方法训练

# utils

一些用到的工具代码

# rpn

本文章一开始的思路和做的一些实验，仅用rpn网络，但结果很差，后续排除了

# rpn_roi_pre

基于faster r-cnn的开集目标定位网络，主要是修改了预测头，引入了集中其他的损失函数



