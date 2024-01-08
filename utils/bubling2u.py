import os
import shutil

save_path = 'F:\\object-detection\\RT-DETR\\RTDETR-main\\dataset/'
split_path = 'D:\\OVD\datasets\\VOCdevkit\\VOC2007\\ImageSets\\Main/'
data_path = 'D:\\OVD\datasets\\VOCdevkit\\VOC2007/'
file_list = ['train','test','val']

for file in file_list:
    with open(split_path+file+'.txt','r') as f:
        img_list = f.readlines()
    img_list = [x.split()[0] for x in img_list]
    img_save_path = save_path+'images/'+file+'/'
    #print(img_save_path)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    label_save_path = save_path+'labels/'+file+'/'
    #print(label_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    
    for img in img_list:
        print(data_path+'JPEGImages/'+img+'.jpg')
        shutil.copy(data_path+'JPEGImages/'+img+'.jpg', save_path+'images/'+file+'/')
        shutil.copy(data_path+'Annotations_1obj_txt/'+img+'.txt', save_path+'labels/'+file+'/')
    