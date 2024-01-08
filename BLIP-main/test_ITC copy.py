from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
from IPython.display import display
import numpy as np
import glob
from data.dataloader_itc import BLIPDataset, frcnn_dataset_collate
from torch.utils.data import DataLoader
from models.blip_itm import blip_itm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('-------------use %s-----------'%(device))
image_size = 384
model_url = 'weights/downstream/model_base_retrieval_coco.pth'
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device='cpu')
class_file = 'voc_classes.txt' 
with open(class_file,'r') as f:
    class_list = f.readlines()
    class_list =[x.split('\n')[0] for x in class_list]
caption = class_list
print('text: %s' %caption)
os.make
img_path = 'D:\\OVD\\datasets\\VOCdevkit\\VOC2007\\JPEGImages/'
ann_folder = 'D:\\OVD\\rpn_roi_pre\\.temp_map_out\\detection-results/'
save_path = ''
data = BLIPDataset(img_path, ann_folder)
print('dataset done!')
gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 0, pin_memory=True,
                    drop_last=True, collate_fn=frcnn_dataset_collate
                    )
for iteration, batch in enumerate(gen):
    crops_batch, confs_batch = batch[0], batch[1]
    for i in range(crops_batch.shape[0]): 
        obj_list = []
        crop = crops_batch[i]
        print(crop.shape)
        with torch.no_grad():
            itc_score = model(crop,caption,match_head='itc').numpy().astype(np.float32)
        #print(itc_score)
        argmax_itc = itc_score.argmax(axis=1)
        #print(type(argmax_itc))
        argmax_itc = argmax_itc.tolist()
        for i in range(len(argmax_itc)):
            print(caption[argmax_itc[i]])



