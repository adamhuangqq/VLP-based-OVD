from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.dataloader_cap import BLIPDataset, frcnn_dataset_collate
from torch.utils.data import DataLoader
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('------------------use %s------------'%(device))
class_file = 'model_data/voc_classes.txt'
with open(class_file,'r') as f:
    class_list = f.readlines()
    class_list =[x.split('\n')[0] for x in class_list]

model_url = 'weights/downstream/model_base_caption_capfilt_large.pth'
image_size = 384
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

img_path = '/home/huangqiqiang/docker/datasets/VOCdevkit/VOC2007/JPEGImages/'
ann_folder = '/home/huangqiqiang/docker/code/rpn_roi_pre/voc_map_out/detection-results_ovd2/'
save_path = '/home/huangqiqiang/docker/code/rpn_roi_pre/voc_map_out/captions/'
if not os.path.exists(save_path):
    os.makedirs(save_path+'caption_file/')
    os.makedirs(save_path+'caption_classes/')

for name in class_list:
    with open(save_path+'caption_classes/'+name+'.txt', 'w') as f:
        pass

num = 0

data = BLIPDataset(img_path, ann_folder)
print('dataset done!')
gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 4, pin_memory=True,
                    drop_last=True, collate_fn=frcnn_dataset_collate
                    )
length = 9271
with tqdm(total=length, desc=f'num 1', postfix=dict, mininterval=0.3) as pbar:
    for iteration, batch in enumerate(gen):
        crops_batch, confs_batch, boxes_batch, file_name_list, obj_list_batch = batch[0], batch[1], batch[2], batch[3], batch[4]
        for i in range(crops_batch.shape[0]):
            caption_list = []
            file_name = file_name_list[i]
            conf = confs_batch[i]
            boxes = boxes_batch[i]
            crop = crops_batch[i].to(device)
            obj_list = obj_list_batch[i]
            with torch.no_grad():
                # beam search
                caption = model.generate(crop, sample=False, num_beams=3, max_length=20, min_length=5)
            file = open(save_path+'caption_file/'+file_name+'.txt', 'w')
            for j in range(len(obj_list)):
                with open(save_path+'caption_classes/'+obj_list[j]+'.txt', 'a') as f:
                    f.write(file_name+' ')
                    f.write(caption[j])
                    f.write('\n')
                file.write(obj_list[j]+' ')
                file.write(caption[j]+'\n')
            file.close()
            pbar.set_postfix(**{'file_name': file_name
                                })
            pbar.update(1)



    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    # print('caption: '+caption[0])
