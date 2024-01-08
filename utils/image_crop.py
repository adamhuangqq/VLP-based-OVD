import os
import numpy as np
import cv2

def file2coor_list(file):
    with open(file,'r') as f:
        coor = f.readlines()
    coor = [x.split()[2:] for x in coor]
    coor = np.array(coor).astype(int)
    #print(coor)
    return coor


def image_crop(save_path, image_path, annotation_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_list = os.listdir(annotation_path)
    for file in file_list:
        coor = file2coor_list(annotation_path+file)
        image = cv2.imread(image_path+file[:-4]+'.jpg')
        for i in range(len(coor)):
            # 这儿的长宽跟正常的xy相反
            corp_img = image[coor[i,1]:coor[i,3], coor[i,0]:coor[i,2], :]
            cv2.imwrite(save_path+file[:-4]+'-'+str(i)+'.jpg', corp_img)
        print('%s has been cropped done!'%(file))





if __name__=='__main__':
    file = 'RPN_results/proposal_select/000001.txt'
    save_path = 'save/'
    image_path = 'JPEGImages/'
    annotation_path = 'RPN_results/proposal_select/'
    image_crop(save_path, image_path, annotation_path)
    os.list