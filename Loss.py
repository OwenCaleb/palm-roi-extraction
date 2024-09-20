# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:59:17 2020

@author: Lim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _neg_loss(pred, gt):

  # 用于目标检测中训练网络，以处理不平衡的正负样本问题。
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)  ground truth
  '''
  pos_inds = gt.eq(1).float()  # 正样本掩码：gt中等于1的位置
  neg_inds = gt.lt(1).float()  # 负样本掩码：gt中小于1的位置
  neg_weights = torch.pow(1 - gt, 4)
  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  num_pos  = pos_inds.float().sum() 
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos 
  return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, pred_tensor, target_tensor):
    return self.neg_loss(pred_tensor, target_tensor)


def _gather_feat(feat, ind, mask=None):
    #angle为例
    dim  = feat.size(2)
    # print(ind.shape) #torch.Size([16, 128])  1000000000000000...0
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # print(ind.shape) #torch.Size([16, 128, 1])
    # print(feat.shape)#torch.Size([16, 16384, 1])
    # print(ind)
    '''
     [[7883],
     [   0],
     [   0],
     ...,
     [   0],
     [   0],
     [   0]],
    '''
    feat = feat.gather(1, ind)#沿着第一维取出ind位置的元素#torch.Size([16, 16384, 1])
    # print(feat.shape) # 16, 128 ,1
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() #torch.Size([16, 128,128,1])
    feat = feat.view(feat.size(0), -1, feat.size(3)) #batchsize width*height channels
    feat = _gather_feat(feat, ind)#提取对应index的特征值
    return feat

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, pred, mask, ind, target):
    pred = _transpose_and_gather_feat(pred, ind)  #(batch_size, num_indices, channels) num_indices 0-width*height-1的索引
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # print(mask.shape) torch.Size([16, 128, 1])
    # print(pred.shape) torch.Size([16, 128, 1])
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4) # 每个目标的平均损失
    return loss

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y
def _relu(x):
    y = torch.clamp(x.relu_(), min = 0., max=179.99)
    return y

class CtdetLoss(torch.nn.Module):
    # loss_weight={'hm_weight':1,'wh_weight':0.1,'reg_weight':0.1}
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.loss_weight = loss_weight
        
    def forward(self, pred_tensor, target_tensor):
        # torch.Size([4, 1, 128, 128]) 'hm': 1, 'wh': 2, 'ang':1, 'reg': 2
        hm_weight = self.loss_weight['hm_weight']
        wh_weight = self.loss_weight['wh_weight']
        reg_weight = self.loss_weight['reg_weight']
        ang_weight = self.loss_weight['ang_weight']
        # 包含多个张量的字典
        # print(pred_tensor.keys()) dict_keys(['hm', 'wh', 'ang', 'reg'])
        # print(target_tensor.keys()) dict_keys(['input', 'hm', 'reg_mask', 'ind', 'wh', 'ang', 'reg'])
        # print(pred_tensor['hm'].size()) torch.Size([16, 1, 128, 128])
        # print(pred_tensor['wh'].size()) torch.Size([16, 2, 128, 128])
        # print(pred_tensor['reg'].size()) torch.Size([16, 2, 128, 128])
        # print(pred_tensor['ang'].size()) torch.Size([16, 1, 128, 128])
        # print(target_tensor['ind'].size()) torch.Size([16, 128])
        # print(target_tensor['reg_mask'].size()) torch.Size([16, 128])
        # print(target_tensor['ang'].size()) torch.Size([16, 128, 1])
        # print(target_tensor['wh'].size()) torch.Size([16, 128, 2])
        hm_loss, wh_loss, off_loss, ang_loss = 0, 0, 0, 0
        # 更平滑
        pred_tensor['hm'] = _sigmoid(pred_tensor['hm'])
#        print(target_tensor['hm'].size())
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm'])
        if ang_weight > 0:
            pred_tensor['ang'] = _relu(pred_tensor['ang'])
            ang_loss += self.crit_wh(pred_tensor['ang'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['ang'])  
        if wh_weight > 0:
            wh_loss += self.crit_wh(pred_tensor['wh'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['wh'])
        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg'], target_tensor['reg_mask'],target_tensor['ind'], target_tensor['reg'])
        return hm_weight * hm_loss + wh_weight * wh_loss + reg_weight * off_loss + ang_weight * ang_loss


