import cv2
import os

def plot_detection_img(image_path, annotation_path, save_path, imgname):

    img = cv2.imread(image_path  + imgname + '.jpg')
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open(annotation_path + imgname + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [x.strip() for x in label][1:]
            label = [float(x) for x in label]            
            pt1 = (int(label[1]), int(label[2]))
            pt2 = (int(label[3]), int(label[4]))
            if label[0]>0.5:
                cv2.putText(img,str(label[0]),pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(img,pt1,pt2,(0,255,0),1)
    cv2.imwrite(save_path + imgname + ".jpg",img)

def plot_gt_img(detection_image_path, gt_annotation_path, save_path, imgname):
    img = cv2.imread(detection_image_path  + imgname + '.jpg')
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open(gt_annotation_path + imgname + ".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [x.strip() for x in label]           
            pt1 = (int(label[1]), int(label[2]))
            pt2 = (int(label[3]), int(label[4]))
            cv2.putText(img,str(label[0]),pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.rectangle(img,pt1,pt2,(0,0,225),1)

    cv2.imwrite(save_path + imgname + ".jpg",img)

if __name__ == '__main__':    
    # image_path = 'D:/OVD/datasets/VOCdevkit/VOC2007/JPEGImages/'
    # annotation_path = 'proposal_select/propasal_select/'
    # gt_annotation_path = 'proposal_select/ground-truth/'
    # save_path = 'proposal_select/plot-gt/'
    # save_path_gt = 'proposal_select/plot-gt-gt/'

    image_path = 'F:\car-detect\dataset\dataset_car\VOC2007\JPEGImages/'
    annotation_path = 'F:\car-detect\dataset\dataset_car\VOC2007\Annotations/'
    gt_annotation_path = 'F:\car-detect\dataset\dataset_car\VOC2007\Annotations_txt/'
    save_path = 'F:\car-detect\dataset\dataset_car\VOC2007/plot-detc/'
    save_path_gt = 'F:\car-detect\dataset\dataset_car\VOC2007/plot-detc/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_gt):
        os.makedirs(save_path_gt)
    pathDir = os.listdir(gt_annotation_path)
    num = 100
    for i in pathDir:
        if num:
            print(i[:-4])
            #plot_detection_img(image_path, annotation_path, save_path, i[:-4])
            plot_gt_img(image_path, gt_annotation_path, save_path_gt, i[:-4])
            num = num - 1
        else:
            break
