import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 



def dice_coeff(y_true, y_pred, batch):
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    smooth = 0.0001  # may change
    if batch:
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
    else:
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
    score = (2. * intersection + smooth) / (i + j + smooth)
    # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return score.mean().numpy()




def m_iou(y_true, y_pred, batch):
    smooth = 0.0001
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    if batch:
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
    else:
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

    score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return score.mean().numpy()

 