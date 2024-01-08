import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import copy

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

#计算两个BBOX的(dx, dy, dw, dh)，即预测和目标的差距
def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

class AnchorTargetCreator(object):
    #n_sample=256, pos_ratio=0.5这两个参数可以调节
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor, gt_label):
        argmax_ious, label, max_iou = self._create_label(anchor, bbox, gt_label)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            #返回值：
            #   loc:所有anchor的调整值
            #   label:所有anchor的label
            return loc, label, max_iou
        else:
            return np.zeros_like(anchor), label, max_iou

    def _calc_ious(self, anchor, bbox):
        #----------------------------------------------#
        #   anchor和bbox的iou，bbox是gt（左下右上）
        #   获得的ious的shape为[num_anchors, num_gt]
        #   即每个anchor都有gt个iou
        #----------------------------------------------#
        ious = bbox_iou(anchor, bbox)

        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #---------------------------------------------------------#
        argmax_ious = ious.argmax(axis=1)
        #---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        #---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        #---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        #   这一步就是实现了防止因为某两个框重合度太高，
        #   使得其中一个框被另一个框完全遮盖
        #---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i
        #---------------------------------------------------------#
        #   返回值：
        #   argmax_ious：每一个先验框最对应的真实框  [num_anchors, ]
        #   max_ious：找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #   gt_argmax_ious：每一个真实框最对应的先验框  [num_gt, ]
        #---------------------------------------------------------#
        return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox, gt_label):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        
        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = gt_label[argmax_ious[max_ious >= self.pos_iou_thresh]] + 1
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label >= 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        #-------------------------------------------------------#
        #   返回值：
        #   argmax_ious：每个先验框对应的最大的真实框的序号[num_anchors,] 
        #   label：每个先验框的标号，（-1，0（背景），各个类别）
        #-------------------------------------------------------#
        return argmax_ious, label, max_ious


class RPNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(RPNTrainer, self).__init__()
        self.model_train    = model_train
        self.optimizer      = optimizer

        self.rpn_sigma      = 1
        self.roi_sigma      = 1

        self.anchor_target_creator      = AnchorTargetCreator()

        self.loc_normalize_std          = [0.1, 0.1, 0.2, 0.2]
    # gt_label:(-1,0,1) 
    # gt_loc:与对应真实框的调整值（也就是pred_loc的真值） 
    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        #取出需要计算损失的框
        pred_loc    = pred_loc[gt_label > 0]
        gt_loc      = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        #计算回归的损失，L1 smooth loss
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            )
        regression_loss = regression_loss.sum()
        num_pos         = (gt_label > 0).sum().float()
        
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss
    
    def forward(self, imgs, bboxes, labels, scale):
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #-------------------------------#
        #   获取公用特征层
        #-------------------------------#
        base_feature = self.model_train(imgs, mode = 'extractor')

        # -------------------------------------------------- #
        #   利用rpn网络获得调整参数、得分、建议框、先验框
        # -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        #   返回值：
        #   rpn_locs：微调的四个坐标
        #   rpn_scores：前景背景得分
        #   rois：感兴趣的区域（经过NMS之后的，一张图片剩600个，最关键的）
        #       roi里面是中心点坐标和长宽的BBOX
        #   roi_indices：表示Roi属于哪一个图片                                     
        #   anchor：原始初始化的anchor，升过维。
        #       （这儿需要强调的是调整是一步到位的，并不是一点点慢慢调）
        #   x = [base_feature, img_size] 在frcnn里面写了
        # ---------------------------------------------------------------------------- #
        rpn_locs, rpn_scores, rpn_fg_scores, rois, roi_indices, roi_locs, roi_scores, anchor = self.model_train(x = [base_feature, img_size], scale = scale, mode = 'rpn')
        
        rpn_loc_loss_all, rpn_cls_loss_all, rpn_reg_loss_all, roi_loc_loss_all, roi_cls_loss_all  = 0, 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels, sample_roi_locs, sample_roi_scores = [], [], [], [], [], []
        for i in range(n):
            bbox        = bboxes[i]#第i张图片的真实框（k，4）
            label       = labels[i]#第i张图片的真实框对应的标签（k，1）
            rpn_loc     = rpn_locs[i]#第i张图片的anchor的微调坐标（anchor数，4）
            rpn_score   = rpn_scores[i]#第i张图片的anchor的前背景分数（anchor数，2）
            rpn_fg_score= rpn_fg_scores[i]#第i张图片的anchor的softmax前景分数（anchor数，1）
            roi         = rois[i]#第i张图片的roi的坐标（600，4）
            roi_loc     = roi_locs[i]#这个变量希望获得roi的微调，用来计算roi的定位损失
            roi_score   = roi_scores[i]#这个变量希望获得roi的预测概率，用来计算roi的分类损失
            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label, gt_reg_label    = self.anchor_target_creator(bbox, anchor[0].cpu().numpy(), label)
            keep                        = np.where(gt_rpn_label > 0)
            gt_reg_label                = gt_reg_label[keep]
            rpn_fg_score                = rpn_fg_score[keep]
            gt_rpn_loc                  = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label                = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
            gt_reg_label                = torch.Tensor(gt_reg_label).type_as(rpn_locs)
            gt_rpn_label_bg             = copy.deepcopy(gt_rpn_label)
            
            gt_rpn_label_bg[gt_rpn_label_bg > 0] = 1
            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            #print(type(rpn_loc),type(gt_rpn_loc))
            
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label_bg, ignore_index=-1)
            if len(rpn_fg_score) > 0:
                rpn_reg_loss = F.mse_loss(rpn_fg_score, gt_reg_label)
            else:
                rpn_reg_loss = 0
            
            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            rpn_reg_loss_all += rpn_reg_loss
        
        a = 0.2    
        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, rpn_reg_loss_all/n]
        losses = losses + [sum(losses)]

        return losses




            
    

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler=None):
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
        return losses




def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
