import cv2
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        #print(np.shape(image))
        return image 
    else:
        image = image.convert('RGB')
        return image 

class BLIPDataset(Dataset):
    def __init__(self, img_path, ann_folder, confidence = 0.5, input_shape = (384, 384), train = False):
        self.file_folder        = ann_folder
        self.img_path           = img_path
        self.file_list          = os.listdir(ann_folder)
        self.input_shape        = input_shape
        self.train              = train
        self.length             = len(self.file_list)
        self.confidence         = confidence

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #print(self.file_list[index])
        file_name = self.file_list[index]
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, conf, box, obj_list  = self.get_crop_img_data(self.file_list[index], self.input_shape)
        return image, conf, box, file_name, obj_list

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def load_demo_image(self, device, image, boxes, image_size=(384,384)):
        transform = transforms.Compose([
            transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        length = boxes.shape[0]
        crop_all = np.zeros((length, 3, image_size[0], image_size[1]))
        for i, box in enumerate(boxes):
            #print(box)
            crop = image.crop([box[0],box[1],box[2],box[3]])
            crop = transform(crop)
            #crop = np.transpose(np.array(crop, dtype=np.float32), (2, 0, 1))
            crop_all[i] = crop

        return crop_all
        

    def get_crop_img_data(self, file, input_shape):
        image = Image.open(self.img_path+file[:-4]+'.jpg').convert('RGB') 
        #image   = cv2.imread(self.img_path+file[:-4]+'.jpg')
        #print(self.img_path+file[:-4]+'.jpg') 
        #print(image.shape)       
        # 获得预测框，并根据置信度进行筛选
        with open(self.file_folder+file,'r') as f:
            ann = f.readlines()
        ann = [x.split() for x in ann]
        ann_list = []
        conf_list = []
        obj_list = []
        #print(ann)
        for x in ann:
            if float(x[1]) > self.confidence:
                ann_list.append(x[2:])
                conf_list.append(x[1])
                obj_list.append(x[0])
            else:
                pass
        # print(ann_list)
        # print(conf_list)
        
        boxes     = np.array([np.array([int(x) for x in box]) for box in ann_list])
        #conf      = np.array([float(x) for x in conf_list])
        crop_all  = self.load_demo_image('cpu', image, boxes)

        return crop_all, conf_list, boxes, obj_list

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    crops = []
    confs = []
    boxes = []
    file_name_list = []
    obj_list = []

    for crop, conf, box, file_name, obj in batch:
        crops.append(crop)
        confs.append(conf)
        boxes.append(box)
        file_name_list.append(file_name)
        obj_list.append(obj)
        
    crops = torch.from_numpy(np.array(crops).astype(np.float32))
    #confs = torch.from_numpy(np.array(confs).astype(np.float32))
    return crops, confs, boxes, file_name_list, obj_list

# if __name__=='__main__':
#     img_path = 'D:\\OVD\\datasets\\VOCdevkit\\VOC2007\\JPEGImages/'
#     ann_folder = '.temp_map_out\\detection-results/'
#     data = BLIPDataset(img_path, ann_folder)
#     print('dataset done!')
#     gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 4, pin_memory=True,
#                                     drop_last=True, collate_fn=frcnn_dataset_collate, 
#                                 )
#     for iteration, batch in enumerate(gen):
#         crops, confs = batch[0], batch[1]
#         for crop in crops:

        