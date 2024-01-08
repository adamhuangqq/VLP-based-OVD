## 本文件夹是在处理数据时所用到的一些工具代码：

具体介绍如下：

2007_train_*.txt：不同数据集在的数据集信息文件，用来作为生成gt文件和裁剪图片的信息

bubling2u.py：数据集转换的代码，是将[Bubbliiiing](https://blog.csdn.net/weixin_44791964) 的代码移植到yolov8的官方代码上的数据集转换

Calculate_result.py：根据预测结果和真实值计算标签匹配的结果，画出混淆矩阵，这个需要有前置处理

proposal_select.py：上行所述的前置处理，得到匹配结果的真值和预测值

car_split.py：将car数据集按照“基类和新类”的规则分为训练集和测试集

voc_split.py：VOC数据集同上

car_dataset.py：将car数据集划分为Bubbliiing代码中的数据集格式

gt-plot_*.py：目标框，预测框的可视化

image_crop.py：一开始的方法，把预测框的区域图像直接裁剪出来，作为BLIP的输入

prompt.py：存放一些propmpt的模板

\*2*.py：标注文件格式转换的文件

map_from_file.py：通过文件计算map（需要先产生预测文件和真实文件，标注格式为非归一化的yolo格式）

### 文件夹的说明：

*_split：数据集的划分，按照Bubbliiing的格式

res：存放混淆矩阵

model_data：存放数据集类别信息

