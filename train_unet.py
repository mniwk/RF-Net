# coing=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from time import time
import os, sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import random 


import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms, datasets

from models.unet import UNet
from models.EMANet import FPN_Net 
from load_data import MyDataset 
from loss import dice_bce_loss 
from metric import dice_coeff 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)



def train_model(train_i):
    batchsize = 4
    i = train_i
    # NAME = 'F51_fold'+str(i+1)+'_UNet'
    # net = UNet(usegaussian=False).cuda()
    # NAME = 'GaPF5_fold'+str(i+1)+'_UNet'
    # net = UNet(usegaussian=True).cuda()
    # NAME = 'EMF5_NOpretrain_fold'+str(i+1)+'_FSPNet'
    # NAME = 'EMF5_fold'+str(i+1)+'_FSPNet'
    NAME = 'F5_fold'+str(i+1)+'_FSPNet'
    net = FPN_Net(is_ema=False).cuda()
    # net.apply(weights_init)
    print(NAME)

    txt_train = 'D163N5fold'+str(train_i+1)+'_train.csv'
    txt_test = 'D163N5fold'+str(train_i+1)+'_test.csv'
    dataset_train = MyDataset(root='/home/wangke/ultrasound_data163/', txt_path=txt_train, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dataset_test = MyDataset(root='/home/wangke/ultrasound_data163/', txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=2)

    mylog = open('models/saved/'+NAME+'.log', 'w')
    total_epoch = 300
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_test_score = 0
    dice_loss = dice_bce_loss()

    for epoch in range(1, total_epoch):
        total_loss = 0
        data_loader_iter = iter(train_loader)
        data_loader_test = iter(test_loader)
        tic = time()
        train_score = 0
        
        net.train()
        for img, mask in data_loader_iter: 
            img = V(img.cuda(), volatile=False)  
            mask_v = V(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            output = net(img)
            loss = dice_loss(mask_v, output)
            loss.backward()
            optimizer.step()
            total_loss += loss
            train_score += dice_coeff(mask, output.cpu().data, False)

        test_score = 0
        test_loss = 0
        

        net.eval()
        with torch.no_grad():   
            for img, mask in data_loader_test:
                # print(img.shape)
                img = V(img.cuda(), volatile=True)  
                # mask_v = V(mask.cuda(), volatile=False)
                output = net(img)
                # test_loss += dice_loss(mask_v, output)
                # print(dice_coeff(mask, output.cpu().data, False))
                test_score += dice_coeff(mask, output.cpu().data, False)

        total_loss = total_loss/len(data_loader_iter)
        train_score = train_score/len(data_loader_iter)
        test_score = test_score/len(data_loader_test)
        # test_loss = test_loss/len(data_loader_test)

        # scheduler.step()

        if test_score>best_test_score:
            best_test_score = test_score
            torch.save(net, 'models/saved/'+NAME+'.pkl')
            print('saved, ', best_test_score, file=mylog, flush=True)
            print('saved, ', best_test_score)


        print('********', file=mylog, flush=True)
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', total_loss.cpu().data.numpy(), 'train_score:', train_score, 'test_score:', test_score, 'best_score is ', best_test_score, file=mylog, flush=True)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', total_loss.cpu().data.numpy(), 'train_score:', train_score, 'test_score:', test_score, 'best_score is ', best_test_score)


def test_model(train_i):
    batchsize = 4
    i = train_i
    unet = torch.load('models/saved/'+NAME+'.pkl')
    txt_test = 'D163N5fold'+str(train_i+1)+'_test.csv'
    dataset_test = MyDataset(root='/home/wangke/ultrasound_data163/', txt_path=txt_test, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False, num_workers=2)




if __name__ == '__main__':
    i = 0
    print('train for fold'+str(i+1))
    train_model(train_i=i)
    print('train for fold'+str(i+1)+'finished')