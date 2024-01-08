from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import numpy as np
import glob
from data.dataloader_file import BLIPDataset, frcnn_dataset_collate
from torch.utils.data import DataLoader
from models.blip_itm import blip_itm
from tqdm import tqdm
from models.net import ITC_net
if __name__ == '__main__':
    classes1111 = [
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
    classes111 = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'dining table.',
        'dog',
        'horse',
        'motorbike',
        'person.',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]
    classes = [
    'car','van','truck','suv','bus',
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-------------use %s-----------'%(device))
    image_size = 384
    model_url = 'weights/downstream/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    ft_net = ITC_net(256,256)
    ft_net_dict      = ft_net.state_dict()
    pretrained_dict = torch.load('weights/ep300-loss0.04471.pth', map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in ft_net_dict.keys() and np.shape(ft_net_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    ft_net_dict.update(temp_dict)
    ft_net.load_state_dict(ft_net_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    ft_net.eval()
    ft_net = ft_net.to(device)

    class_file = 'model_data/car_classes.txt'
    with open(class_file,'r') as f:
        class_list = f.readlines()
        class_list =[x.split('\n')[0] for x in class_list]
    caption = class_list
    print('text: %s' %caption)

    img_path = 'C:\\datasets\\dataset_car\\VOC2007\\JPEGImages/'
    ann_folder = 'C:\\code\\proposal\\confusion_matrix\\val_conf_gt/'
    save_path = 'C:\\code\\proposal\\confusion_matrix\\val_conf_gt_blip/'
    
    data = BLIPDataset(img_path, ann_folder)
    print('dataset done!')
    gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 8, pin_memory=True,
                        drop_last=True, collate_fn=frcnn_dataset_collate
                        )
    num = 0

    def file_write(obj_list, conf, boxes, save_path, file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path+file_name, 'w')
        for i in range(len(obj_list)):
            f.write(obj_list[i]+' ')
            f.write(str(conf[i])+' ')
            f.write(str(boxes[i][0])+' '+str(boxes[i][1])+' '+str(boxes[i][2])+' '+str(boxes[i][3])+'\n')
        f.close()

    def file_write_more_inf(obj_list, conf, boxes, save_path, file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path+file_name, 'w')
        for i in range(len(obj_list)):
            f.write(obj_list[i][0]+obj_list[i][1]+obj_list[i][2])
            f.write(str(conf[i])+' ')
            f.write(str(boxes[i][0])+' '+str(boxes[i][1])+' '+str(boxes[i][2])+' '+str(boxes[i][3])+'\n')
        f.close()

    length = 2741

    with tqdm(total=length, desc=f'num 1', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            num += 1
            #print(num)
            crops_batch, confs_batch, boxes_batch, file_name_list = batch[0], batch[1], batch[2], batch[3]
            for i in range(crops_batch.shape[0]):
                obj_list = []
                file_name = file_name_list[i]
                conf = confs_batch[i]
                boxes = boxes_batch[i]
                crop = crops_batch[i].to(device)
                with torch.no_grad():
                    image_feat, text_feat, _ = model(crop,classes,match_head='itc')
                text_feat = text_feat.cpu()
                with torch.no_grad():
                    image_feat_ft = ft_net(image_feat).cpu()
                    itc_score = (image_feat_ft @ text_feat.t()).numpy().astype(np.float32)

                argmax_itc = itc_score.argmax(axis=1)
                
                argsort_itc = itc_score.argsort(axis=1)
                argmax_itc = argmax_itc.tolist()
                for i in range(len(argmax_itc)):
                    
                    if True:#itc_score[i][argmax_itc[i]] > 0:
                        obj_list.append(caption[argmax_itc[i]])
                    else:
                        obj_list.append('object')
                file_write(obj_list, conf, boxes, save_path, file_name)
                #print(file_name)
                # for i in range(argsort_itc.shape[0]):
                #     obj_list.append([])
                #     for j in range(3):
                #         k = -1 - j
                #         obj_list[i].append(caption[argsort_itc[i][k]]+' '+str(itc_score[i][argsort_itc[i][k]])+' ')
                # file_write_more_inf(obj_list, conf, boxes, save_path, file_name)

            pbar.set_postfix(**{'file_name': file_name
                                })
            pbar.update(1)