import numpy as np

def frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        print(boxes)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

# input_shape = (300,300)
# image_shape = (100,200)

# roi = np.array([[30.0,120.0,60.0,150.0]])
# roi[..., [0, 2]] = (roi[..., [0, 2]]) / input_shape[1]
# roi[..., [1, 3]] = (roi[..., [1, 3]]) / input_shape[0]
# print(roi) 
# box_xy, box_wh = (roi[:, 0:2] + roi[:, 2:4])/2, roi[:, 2:4] - roi[:, 0:2]
# print(box_wh,box_xy)

# print(frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape))
# gt_txt = '2007_val.txt'
# with open(gt_txt, encoding='utf-8') as f:
#         file_inf = f.readlines()
#         file_inf = [x.split() for x in file_inf]
# image_ids = [x[0].split('/')[-1][:-4] for x in file_inf]
# bboxes = [x[1:] for x in file_inf]
# print(bboxes)

a='1234'
b='x123erfre'
a=b
print(a)
