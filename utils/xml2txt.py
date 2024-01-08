import xml.etree.ElementTree as ET
import os
from os import getcwd
from os.path import join
import glob

sets = ['train','test']#分别保存训练集和测试集的文件夹名称
classes = ['car', 'van', 'truck', 'suv', 'bus']#标注时的标签

#-------------------------------------------------------
#   xml中框的左上角坐标和右下角坐标(x1,y1,x2,y2)
#   txt中的中心点坐标和宽和高(x,y,w,h)，并且归一化
#-------------------------------------------------------
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
#-------------------------------------------------------
#   将xml标签转换成txt标签，但是依然用最大最小坐标表示
#-------------------------------------------------------    
def convert_annotation(data_dir,save_dir):
    in_file = open(data_dir) #读取xml
    out_file = open(save_dir, 'w') #保存txt
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    #w = int(size.find('width').text)
    #h = int(size.find('height').text)
    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #if cls not in classes or int(difficult) == 1:
        #    continue
        cls_id = 0#classes.index(cls)#获取类别索引
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        #print(b)
        bb = b
        out_file.write(str(cls_id) + " " + " ".join([str('%d'%a) for a in bb]) + '\n')


data_dir='D:\\OVD\\datasets\\VOCdevkit\\VOC2007\\Annotations/'
save_dir='D:\\OVD\\datasets\\VOCdevkit\\VOC2007\\Annotations_1obj_txt/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_list = os.listdir(data_dir)
num = 0 
for i in file_list:
    convert_annotation(data_dir+i,save_dir+i[:-3]+'txt')
print("Done!!!")


