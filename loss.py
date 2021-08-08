import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lovasz_losses import lovasz_hinge

import cv2
import numpy as np
class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b


class iou_loss1(nn.Module):
    def __init__(self, batch=True):
        super(iou_loss1, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_iou_coeff(self, y_true, y_pred):
        smooth = 1e-4  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (intersection + smooth) / (i + j + smooth-intersection)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_iou_loss(self, y_true, y_pred):
        loss = 1 - self.soft_iou_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_iou_loss(y_true, y_pred)
        return a+b



class FocalLoss(nn.Module):
    def __init__(self, batch=False):
        super(FocalLoss, self).__init__()
        self.batch = batch

    def focalLoss(self, y_true, y_pred, weight, gamma=1):
        if y_pred.dim()>2:
            y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
            y_pred = y_pred.transpose(1,2)
            y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
            y_true = y_true.transpose(1,2)
        if self.batch:
            y_pred = y_pred.contiguous().view(-1,y_pred.size(2))
            y_true = y_true.contiguous().view(-1,y_true.size(2))
            loss_value = -y_true*(1-y_pred)**gamma*torch.log(y_pred)-(1-y_true)*y_pred**gamma*torch.log(1-y_pred)
        else:
            weight = weight.unsqueeze(1).unsqueeze(1)
            positive = -(1-y_pred)**gamma *torch.log(y_pred+1e-2) *y_true
            negative = -y_pred**gamma *torch.log(1-y_pred+1e-2) *(1-y_true)
            # print(weight.size(), positive.size())
            loss_value = positive*weight + negative*(1-weight)
        
        return loss_value.mean()

    def __call__(self, y_pred, y_true, gamma=1):
        y_true[y_true>=0.5]=1
        y_true[y_true<0.5]=0
        i = y_true.sum(1).sum(1).sum(1)
        i_ = (1-y_true).sum(1).sum(1).sum(1)
        weight = torch.div(i_, i+i_)
        return self.focalLoss(y_true, y_pred, weight, gamma=gamma)
    


class iou_loss2(nn.Module):
    def __init__(self, batch=True):
        super(iou_loss2, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        # self.weight = 0

    def focalLoss(self, y_true, y_pred, gamma=1):
        if y_pred.dim()>2:
            y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
            y_pred = y_pred.transpose(1,2)
            y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
            y_true = y_true.transpose(1,2)
        if self.batch:
            y_pred = y_pred.contiguous().view(-1,y_pred.size(2))
            y_true = y_true.contiguous().view(-1,y_true.size(2))
            loss_value = -y_true*(1-y_pred)^gamma*torch.log(y_pred)-(1-y_true)*y_pred^gamma*torch.log(1-y_pred)
        else:
            weight = self.weight.unsqueeze(1).unsqueeze(1)
            positive = -(1-y_pred)**gamma *torch.log(y_pred+1e-2) *y_true
            negative = -y_pred**gamma *torch.log(1-y_pred+1e-2) *(1-y_true)
            # print(weight.size(), positive.size())
            loss_value = positive*weight + negative*(1-weight)
        
        return loss_value.mean()

    def soft_iou_coeff(self, y_true, y_pred):
        smooth = 1e-4  # may change
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        i_ = (1-y_true).sum(1).sum(1).sum(1)
        j_ =  (1-y_pred).sum(1).sum(1).sum(1)
        intersection2 = ((1-y_true) * (1-y_pred)).sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        self.weight = torch.div(i_, i+i_)
        score = (intersection + smooth) / (i + j + smooth-intersection)*self.weight+(1-self.weight)*(intersection2+smooth)/(i_ + j_ + smooth-intersection2)
        return score.mean()

    def soft_iou_loss(self, y_true, y_pred):
        loss = 1 - self.soft_iou_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_iou_loss(y_true, y_pred)
        # a2 = self.focalLoss(y_true, y_pred)
        # a = a1*0.5+0.5*a2
        return b





def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a+b


class dice_bce_loss2(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss2, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
            i2 = (1-y_true).sum(1).sum(1).sum(1)
            j2 = (1-y_pred).sum(1).sum(1).sum(1)
            intersection2 = ((1-y_true) * (1-y_pred)).sum(1).sum(1).sum(1)
            weight = torch.div(i2, i+i2)
        score = weight*(2. * intersection + smooth) / (i + j + smooth) + (1-weight)*(2. * intersection2 + smooth) / (i2 + j2 + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return 0.5*a+b





import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss



class ltg_loss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        bceloss = nn.BCELoss()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        loss = 0
        input_ = input.view(N, -1)
        target_ = target.view(N, -1)



class edge_penelize(nn.Module):
    def __init__(self):
        super(edge_penelize, self).__init__()
        bceloss = nn.BCELoss()

    def __call__(self, input, target):
        # 1. 腐蚀+扩张，得到边缘
        # or 直接读取边缘   省cpu
        input_edge = input*target 
        
