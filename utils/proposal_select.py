#-------------------------------------------------------------------#
#   从预测的proposal中找到和g重t合度比较高的作为后续分类的建议框
#-------------------------------------------------------------------#
import numpy as np
import os
import torch
from utils.utils import get_classes
obj_list_voc = [
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
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
obj_list = [
   'car','van','truck','suv','bus',
]
#-------------------------------------------------------------------#
#   重写gt，因为在map计算中写的gt没有类别属性
#-------------------------------------------------------------------#
def gt_file(save_path, gt_txt, class_names):
    with open(gt_txt, encoding='utf-8') as f:
        file_inf = f.readlines()
        file_inf = [x.split() for x in file_inf]
    
    image_ids = [x[0].split('/')[-1][:-4] for x in file_inf]
    bboxes = [x[1:] for x in file_inf]
    for image_id,bbox in zip(image_ids, bboxes):
        with open(os.path.join(save_path, "ground-truth/"+image_id+".txt"), "w") as new_f:        
            for box in bbox:
                left, top, right, bottom, obj = box.split(',')
                #print(left, top, right, bottom, type(obj))
                obj_name = class_names[int(obj)]
                new_f.write("%s %s %s %s %s %s\n" % (obj_name, '1.00', left, top, right, bottom))
        print('%s write done!' %(image_id))
    print('All files write done!')
    return

#计算IOU，计算的是两组bbox的iou
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def file2bbox(file):
    with open(file,'r') as f:
        bbox = f.readlines()
        bbox = [x.split() for x in bbox]
    for box in bbox:
        if box[0] == 'object':
            box[0] += 'allthings' 
    if len(bbox)>0:
        return np.array(bbox)
    else:
        return np.array(bbox)

def propasal_select(gt_path, detc_path, save_path, IOU = 0.5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_list = os.listdir(gt_path)
    num = 0
    for file in file_list:
        #num+=1
        if num>10: break
        print(file)
        gt_bbox = file2bbox(gt_path+file)
        #print(gt_bbox)
        detc_bbox = file2bbox(detc_path+file)
        gt_bbox_coor = gt_bbox[:,1:].astype(np.float32)
        if len(detc_bbox) > 0:
            detc_bbox_coor = detc_bbox[:,2:].astype(np.float32)
        else:
            with open(save_path+file,'w') as w:
                w.write('')
                continue
        ious = bbox_iou(detc_bbox_coor,gt_bbox_coor)
        argmax_ious = ious.argmax(axis=1)
        max_ious = np.max(ious, axis=1)
        keep = np.where(max_ious>=IOU)
        if len(detc_bbox) > 0:
            argmax_ious = argmax_ious[keep]
            max_ious = max_ious[keep]
            #detc_bbox = detc_bbox[keep]
            detc_bbox[keep,0] = gt_bbox[argmax_ious,0]
            #print(detc_bbox[:,0])
            with open(save_path+file,'w') as w:
                for box in detc_bbox:
                    obj, score, left, top, right, bottom = box
                    if obj in obj_list and obj != 'suv':
                        w.write("%s %s %s %s %s %s\n" % (obj, score, left, top, right, bottom))
        else:
            with open(save_path+file,'w') as w:
                w.write('')
                continue
            
        #print(argmax_ious)
        



if __name__ == '__main__':
    #第一步，写gt
    save_path  = 'proposal_select/gt/val_conf/'
    if not os.path.exists(save_path+'ground-truth/'):
        os.makedirs(save_path+'ground-truth/')
    gt_txt = '2007_val.txt'
    class_path = 'model_data/car_classes.txt'
    class_names,_ = get_classes(class_path)
    #print(type(class_names))
    gt_file(save_path, gt_txt, class_names)
    # #第二步，找到最符合的proposal
    # gt_path = 'proposal_select/car/ground-truth/'
    # detc_path = 'proposal_select/car/rpn_toi_pre_results/detection-results_ovd/'
    # save_path = 'proposal_select/car/rpn_toi_pre_results/detection-results_obj_gt_nosuv/'
    # propasal_select(gt_path, detc_path, save_path)






            

