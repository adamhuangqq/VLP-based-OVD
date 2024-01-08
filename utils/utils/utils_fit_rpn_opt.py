import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

#------------------------------------------------------------------#
#   model：FasterRCNN
#   train_util：FasterRCNNTrainer
#   loss_history：存储损失
#   eval_callback：记录评价指标
#   optimizer：优化器
#   epoch：第几个世代数
#   epoch_step：每个世代分为多少个batchsize
#   epoch_step_val：验证集的
#   gen：数据迭代器
#   gen_val：验证集数据迭代器
#   Epoch：总共的世代数
#   fp16, scaler, save_period, save_dir：这些都是训练策略，并不重要
#------------------------------------------------------------------#
def fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    rpn_reg_loss = 0
    
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
            # 返回的losses包括这几部分
            rpn_loc, rpn_cls, rpn_reg, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler)
            total_loss      += total.item()
            rpn_loc_loss    += rpn_loc.item()
            rpn_cls_loss    += rpn_cls.item()
            rpn_reg_loss    += rpn_reg.item()
           
            
            pbar.set_postfix(**{'total_loss'    : total_loss * 10 / (iteration + 1), 
                                'rpn_loc'       : rpn_loc_loss * 10 / (iteration + 1),  
                                'rpn_cls'       : rpn_cls_loss * 10 / (iteration + 1),
                                'rpn_reg'       : rpn_reg_loss * 10 / (iteration + 1),                
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

                train_util.optimizer.zero_grad()
                _, _, val_total = train_util.forward(images, boxes, labels, 1)
                val_loss += val_total.item()
                
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
