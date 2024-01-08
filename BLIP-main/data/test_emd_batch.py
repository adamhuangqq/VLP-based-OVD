from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import numpy as np
import glob
from data.dataloader_emd import BLIPDataset, frcnn_dataset_collate
from torch.utils.data import DataLoader
from models.blip_itm import blip_itm
from tqdm import tqdm
if __name__ == '__main__':
    classes = [
        'a close up of aeroplane',
        'a close up of bicycle',
        'a close up of bird',
        'a close up of boat',
        'a close up of bottle',
        'a close up of bus',
        'a close up of car',
        'a close up of cat',
        'a close up of chair',
        'a close up of cow',
        'a close up of a dining table.',
        'a close up of dog',
        'a close up of horse',
        'a close up of motorbike',
        'a close up of a person.',
        'a close up of a potted plant',
        'a close up of sheep',
        'a close up of sofa',
        'a close up of train',
        'a close up of tvmonitor',
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-------------use %s-----------'%(device))

    image_size = 384
    model_url = 'weights/downstream/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    class_file = 'model_data/voc_classes.txt'
    with open(class_file,'r') as f:
        class_list = f.readlines()
        class_list =[x.split('\n')[0] for x in class_list]
    caption = class_list

    print('text: %s' %caption)

    img_path = 'D:\\OVD\\datasets\\VOCdevkit\\VOC2007\\JPEGImages/'
    ann_folder = 'D:\\OVD\\BLIP-main\\dataset\gt\\train/'
    save_path = 'D:\\OVD\\BLIP-main\\dataset\gt-emd\\train/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in class_list:
        with open(save_path+name+'.txt', 'w') as f:
            pass
    data = BLIPDataset(img_path, ann_folder)

    print('dataset done!')

    gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 8, pin_memory=True,
                        drop_last=True, collate_fn=frcnn_dataset_collate
                        )
    num = 0

    def file_write(emds, objs, save_path):
        for i in range(len(objs)):
            with open(save_path+objs[i]+'.txt','a') as f:
                tensor_string = ' '.join([str(item) for item in emds[i].tolist()])
                f.write(tensor_string)
                f.write('\n')

    length = 9271

    with tqdm(total=length, desc=f'num 1', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            num += 1
            #print(num)
            crops_batch, objs_batch = batch[0], batch[1]
            for i in range(crops_batch.shape[0]):
                objs = objs_batch[i]
                crops = crops_batch[i].to(device)
                with torch.no_grad():
                    image_feat, text_feat, argmax_itc = model(crops,classes,match_head='itc')
                image_feat = image_feat.cpu()
                text_feat = text_feat.cpu()
                file_write(image_feat, objs, save_path)
                #print(file_name)

            pbar.set_postfix(**{'num': num
                                })
            pbar.update(1)
            




