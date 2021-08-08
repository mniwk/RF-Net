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
import pandas as pd 
# import random 


import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import torch.nn.functional as F 
from torchvision import transforms, datasets

# from models.unet import UNet
from models.LTGNet import UNet 
from models.gt_guide import RebackNet 
from models.reback import _Res34_unet, _nonlocal_unet, _Reback_v1, _Reback_v2
from load_data import MyDataset, MyDataset2 
from loss import dice_bce_loss, dice_bce_loss2, iou_loss1, iou_loss2, LovaszHingeLoss
from metric import dice_coeff, m_iou 
from models.ssim import SSIM 
torch.set_num_threads(10)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"



# recall
def sensitive(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fn = (gt*(1-pr)).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fn+0.01) 

    return score.mean().numpy()


def precision(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fp = ((1-gt)*pr).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fp+0.01)

    return score.mean().numpy()


# recall1
def sensitive1(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fn = (gt*(1-pr)).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fn+0.01) 

    return score


def precision1(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    fp = ((1-gt)*pr).sum(1).sum(1).sum(1)
    score = (tp+0.01)/(tp+fp+0.01)

    return score


def accuracy(gt, pr):
    tp = (pr*gt).sum(1).sum(1).sum(1)
    tn = ((1-pr)*(1-gt)).sum(1).sum(1).sum(1)
    fp = (pr*(1-gt)).sum(1).sum(1).sum(1)
    fn = ((1-pr)*gt).sum(1).sum(1).sum(1)
    score = (tp+tn+0.01)/(tp+tn+fp+fn+0.01)

    return score.mean().numpy()


def specificity(gt, pr):
    tn = ((1-pr)*(1-gt)).sum(1).sum(1).sum(1)
    fp = (pr*(1-gt)).sum(1).sum(1).sum(1)
    score = (tn+0.01)/(tn+fp+0.01)

    return score.mean().numpy()


def f1_score(gt, pr):
    precision = precision1(gt, pr)
    sensitive = sensitive1(gt, pr)
    score = 2*(precision*sensitive)/(precision+sensitive)

    return score.mean().numpy()





def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

# 依次计算每一个iteration 轮次的结果， 首先试一下循环1次， 共需要t=2/(4)
# 1. 先是计算了
def compute_loss(pre_loss, pre_final, mask):
    loss_all = 0
    dice_loss = dice_bce_loss2(batch=False)
    ssim_loss = SSIM(window_size=11, size_average=True)
    bce_loss = nn.BCELoss(size_average=True)
    mask_d1 = F.interpolate(mask, size=[128, 128], mode='nearest')
    mask_d2 = F.interpolate(mask, size=[64, 64], mode='nearest')
    mask_d3 = F.interpolate(mask, size=[32, 32], mode='nearest')
    for i in range(len(pre_loss)): 
        loss1 = dice_loss(mask, pre_loss[i][0]) 
        # loss2 = dice_loss(mask_d1, pre_loss[i][1]) 
        # loss3 = dice_loss(mask_d2, pre_loss[i][2]) 
        # loss4 = dice_loss(mask_d3, pre_loss[i][3]) 
        loss2 = bce_loss(pre_loss[i][1], mask_d1)
        loss3 = bce_loss(pre_loss[i][2], mask_d2)
        loss4 = bce_loss(pre_loss[i][3], mask_d3)
        # loss2 = dice_loss(mask, F.interpolate(pre_loss[i][1], size=[256, 256], mode='bilinear')) 
        # loss3 = dice_loss(mask, F.interpolate(pre_loss[i][2], size=[256, 256], mode='bilinear')) 
        # loss4 = dice_loss(mask, F.interpolate(pre_loss[i][3], size=[256, 256], mode='bilinear')) 
        # loss1 = dice_loss(mask, pre_loss[i][0]) \
        #     + 1 - ssim_loss(pre_loss[i][0], mask)
        # loss2 = dice_loss(mask, F.interpolate(pre_loss[i][1], size=[256, 256], mode='bilinear')) \
        #     + 1 - ssim_loss(F.interpolate(pre_loss[i][1], size=[256, 256], mode='bilinear'), mask)
        # loss3 = dice_loss(mask, F.interpolate(pre_loss[i][2], size=[256, 256], mode='bilinear')) \
        #     + 1 - ssim_loss(F.interpolate(pre_loss[i][2], size=[256, 256], mode='bilinear'), mask)
        # loss4 = dice_loss(mask, F.interpolate(pre_loss[i][3], size=[256, 256], mode='bilinear')) \
        #     + 1 - ssim_loss(F.interpolate(pre_loss[i][3], size=[256, 256], mode='bilinear'), mask)
        loss_i = loss1+loss2+loss3+loss4
        # loss_i = loss1
        loss_all += loss_i
    
    loss_final = dice_loss(mask, pre_final)
    
    return loss_all, loss_final
        

def train_model(train_i, data_i, threshold, order, test_data1, test_data2):
    # data_i for training and validation, test_data for testing, 测试的话此数据集就全部用作测试，用all_data
    batchsize = 16
    i = train_i
    # net = _Res34_unet().cuda()
    net = _Reback_v2().cuda()
    # net = _nonlocal_unet().cuda()
    netname = "res"

    if data_i == -1:
        epoch_num = 100
        txt_train = 'FuseDatafold'+str(train_i+1)+'_train.csv'
        NAME = 'D2/D23_Res34_unet_'+netname+str(i+1)+'_'+str(order)
        # NAME = 'D2/D23_Res34_unet_'+netname+str(i+1)+'_'+str(order)+'1'
        NAME2 = 'D2/inter_pic/'
        print(NAME)
        dataset_train = MyDataset(root='/home/wangke/ultrasound_data2/', txt_path=txt_train, lab_pics=['fuse_data_pic/', 'fuse_data_lab/', 'inter_pic/'], istrain=True, transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)
        txt_validate = 'FuseDatafold'+str(train_i+1)+'_test.csv'
        dataset_validate = MyDataset(root='/home/wangke/ultrasound_data2/', txt_path=txt_validate, lab_pics=['fuse_data_pic/', 'fuse_data_lab/', 'inter_pic/'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)


    if test_data1 == 205:
        txt_test = 'all_data.csv'
        dataset_test1 = MyDataset(root='/home/wangke/ultrasound_data163/', txt_path=txt_test, lab_pics=['process_pic_163/', 'process_lab_163/'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)
    if test_data2 == 205:
        txt_test = 'all_data.csv'
        dataset_test2 = MyDataset(root='/home/wangke/ultrasound_data163/', txt_path=txt_test, lab_pics=['process_pic_163/', 'process_lab_163/'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)
    

    if test_data1 == 4:  
        txt_test = 'all_data2.csv'
        dataset_test1 = MyDataset(root='/home/wangke/ultrasound_data4/', txt_path=txt_test, lab_pics=['data_pic/', 'data_lab/'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)
    if test_data2 == 4:  
        txt_test = 'all_data2.csv'
        dataset_test2 = MyDataset(root='/home/wangke/ultrasound_data4/', txt_path=txt_test, lab_pics=['data_pic/', 'data_lab/'], transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), pre_num=0)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
    validate_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    test1_loader = torch.utils.data.DataLoader(dataset_test1, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    test2_loader = torch.utils.data.DataLoader(dataset_test2, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

    num_no = 0
    max_numno = 5
    mylog = open('models/saved/'+NAME+'.log', 'w')
    total_epoch = epoch_num
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4, amsgrad=True, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_validate_score = 0
    best_validate_loss = 999
    # loss_loss = LovaszHingeLoss()
    # loss_loss = dice_bce_loss2(batch=False)
    loss_loss = iou_loss2(batch=False)
    bce_loss = nn.BCELoss(size_average=True)
    save_mid = 0
    # ssim_loss = SSIM(window_size=11, size_average=True)

    
    train_loss = []
    test_loss2 = []
    test2_loss2 = []
    validate_loss2 = []
    train_dice = []
    test_dice = []
    test2_dice = []
    validate_dice = []
    train_miou = []
    test_miou = []
    test2_miou = []
    validate_miou = []
    train_sc = 0
    for epoch in range(1, total_epoch+1):
        total_loss = 0
        loss_final_all = 0
        data_loader_iter = iter(train_loader)
        data_loader_validate = iter(validate_loader)
        data_loader_test = iter(test1_loader)
        data_loader_test2 = iter(test2_loader)
        tic = time()
        train_score = 0
        train_miou_b = 0
        
        isneedres = False

        net.train()
        for img, mask, id_img in data_loader_iter: 
            # print(id_img)
            img = V(img.cuda(), requires_grad=False)  
            mask_v = V(mask.cuda(), requires_grad=False)
            # res_v = V(res.cuda(), requires_grad=False)
            
            optimizer.zero_grad()
            # out_final, out_res = net(img, isneedres)
            # out_final, out_res = net(img)
            out_final2, out_res, out_final = net(img)
            # loss_all = dice_loss(mask_v, out_final)
            loss_all = loss_loss(mask_v, out_final)
            loss_step = loss_all
            loss_step += loss_loss(mask_v, out_final2)
            # print(train_sc-threshold)
            # 
            # if train_sc>threshold:
            res_lab = torch.abs(torch.add(mask_v, torch.neg(out_final)))
            res_lab = res_lab.detach()
                # if save_mid==1:
                #     for i, id_i in enumerate(id_img):
                #         cv2.imwrite('models/saved/'+NAME2+id_i, res_lab[i,0,:,:].cpu().data.numpy()*255)
                #         # plt.imsave('models/saved/'+NAME2+id_i+'.png', res_lab[i,0,:,:].cpu().data.numpy(), cmap='gray')    
            loss_res = bce_loss(F.interpolate(out_res, size=(256, 256), mode='bilinear'), res_lab)
            # if loss_res<0.01:isneedres=True
            loss_step +=loss_res

            # loss_all, loss_final = compute_loss(out_loss, out_final, mask_v)
            loss_step.backward()
            optimizer.step()
            total_loss += loss_all
            # loss_final_all += loss_final
            train_score += dice_coeff(mask, out_final.cpu().data, False)
            train_miou_b += m_iou(mask, out_final.cpu().data, False)

        total_loss = total_loss.cpu().data.numpy()/len(data_loader_iter)
        train_score = train_score/len(data_loader_iter)
        train_miou_b = train_miou_b/len(data_loader_iter)
        

        test_score = 0
        test_loss = 0
        test_miou_b = 0
        test2_score = 0
        test2_loss = 0
        test2_miou_b = 0
        test_final_loss = 0
        validate_score = 0
        validate_loss = 0
        validate_miou_b = 0
        

        net.eval()
        with torch.no_grad():   
            for img, mask, id_img in data_loader_test:
                img = V(img.cuda(), requires_grad=False)  
                mask_v = V(mask.cuda(), requires_grad=False)
                # out_final, out_res = net(img, isneedres)
                # out_final, out_res = net(img)
                out_final2, out_res, out_final = net(img)
                loss_all = loss_loss(mask_v, out_final)
                # loss_all += loss_loss(mask_v, out_1)
                test_loss += loss_all
                test_score += dice_coeff(mask, out_final.cpu().data, False)
                test_miou_b += m_iou(mask, out_final.cpu().data, False)

            for img, mask, id_img in data_loader_test2:
                img = V(img.cuda(), requires_grad=False)  
                mask_v = V(mask.cuda(), requires_grad=False)
                # out_final, out_res = net(img, isneedres)
                # out_final, out_res = net(img)
                out_final2, out_res, out_final = net(img)
                # loss_all += loss_loss(mask_v, out_1)
                loss_all = loss_loss(mask_v, out_final)
                test2_loss += loss_all
                test2_score += dice_coeff(mask, out_final.cpu().data, False)
                test2_miou_b += m_iou(mask, out_final.cpu().data, False)

            for img, mask, id_img in data_loader_validate:
                img = V(img.cuda(), requires_grad=False)  
                mask_v = V(mask.cuda(), requires_grad=False)
                # res_v = V(res.cuda(), requires_grad=False)
                # out_final, out_res = net(img, isneedres)
                # out_final, out_res = net(img)
                out_final2, out_res, out_final = net(img)
                loss_all = loss_loss(mask_v, out_final)
                # if train_sc>threshold:
                #     # if save_mid==1:
                #     #     for i, id_i in enumerate(id_img):
                #     #         # plt.imsave('models/saved/'+NAME2+id_i+'.png', res_lab[i,0,:,:].cpu().data.numpy(), cmap='gray')   
                #     #         cv2.imwrite('models/saved/'+NAME2+id_i, res_lab[i,0,:,:].cpu().data.numpy()*255) 
                # res_lab = torch.abs(torch.add(mask_v, torch.neg(out_final)))
                # loss_res = bce_loss(out_res, res_lab)
                # loss_all += loss_res
                validate_loss += loss_all
                validate_score += dice_coeff(mask, out_final.cpu().data, False)
                validate_miou_b += m_iou(mask, out_final.cpu().data, False)

        

        validate_score = validate_score/len(data_loader_validate)
        validate_loss = validate_loss.cpu().data.numpy()/len(data_loader_validate)
        validate_miou_b = validate_miou_b/len(data_loader_validate)
        test_score = test_score/len(data_loader_test)
        test_loss = test_loss.cpu().data.numpy()/len(data_loader_test)
        test_miou_b = test_miou_b/len(data_loader_test)
        test2_score = test2_score/len(data_loader_test2)
        test2_loss = test2_loss.cpu().data.numpy()/len(data_loader_test2)
        test2_miou_b = test2_miou_b/len(data_loader_test2)

        train_sc = validate_score
        if train_sc>threshold:
            # if save_mid==2: save_mid=-1
            save_mid+=1
        # print(train_sc, save_mid)

        train_loss.append(total_loss)
        test_loss2.append(test_loss)
        test2_loss2.append(test2_loss)
        validate_loss2.append(validate_loss)

        train_dice.append(train_score)
        test_dice.append(test_score)
        test2_dice.append(test2_score)
        validate_dice.append(validate_score)

        train_miou.append(train_miou_b)
        test_miou.append(test_miou_b)
        test2_miou.append(test2_miou_b)
        validate_miou.append(validate_miou_b)

        scheduler.step()

        # if validate_score>best_test_score:
        #     best_validate_score = validate_score
        #     torch.save(net, 'models/saved/'+NAME+'.pkl')
        #     print('saved, ', best_validate_score, file=mylog, flush=True)
        #     print('saved, ', best_validate_score)
        if validate_loss<best_validate_loss:
            best_validate_loss = validate_loss
            torch.save(net, 'models/saved/'+NAME+'.pkl')
            print('saved, ', best_validate_loss, file=mylog, flush=True)
            print('saved, ', best_validate_loss)
            num_no =0
        else:
            num_no +=1
        
        if num_no>=max_numno:
            num_no =0
            net = torch.load('models/saved/'+NAME+'.pkl')
            print('loaded, ', best_validate_loss, file=mylog, flush=True)
            print('loaded, ', best_validate_loss)


        print('********', file=mylog, flush=True)
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', total_loss, 'train_score:', train_score, 'train_miou:', train_miou_b, 'validate_loss:', validate_loss, 'validate_score:', validate_score, 'validate_miou:', validate_miou_b, 'test1_loss:', test_loss, 'test1_score:', test_score, 'test1_miou:', test_miou_b, 'test2_loss:', test2_loss, 'test2_score:', test2_score, 'test2_miou:', test2_miou_b, file=mylog, flush=True)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic), 'train_loss:', total_loss, 'train_score:', train_score, 'train_miou:', train_miou_b, 'validate_loss:', validate_loss, 'validate_score:', validate_score, 'validate_miou:', validate_miou_b, 'test1_loss:', test_loss, 'test1_score:', test_score, 'test1_miou:', test_miou_b, 'test2_loss:', test2_loss, 'test2_score:', test2_score, 'test2_miou:', test2_miou_b)

    # plot
    # loss
    plt.figure()
    epochs = range(total_epoch)
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, validate_loss2, 'c', label='validate_loss')
    plt.plot(epochs, test_loss2, 'r', label='test1_loss')
    plt.plot(epochs, test2_loss2, 'g', label='test2_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig('models/saved/'+NAME+'_loss.png')

    plt.figure()
    plt.plot(epochs, train_dice, 'b', label='train_dice')
    plt.plot(epochs, validate_dice, 'c', label='validate_dice')
    plt.plot(epochs, test_dice, 'r', label='test1_dice')
    plt.plot(epochs, test2_dice, 'g', label='test2_dice')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.legend(loc="lower right")
    plt.savefig('models/saved/'+NAME+'_dice.png')



def test_model(train_i, model_path, chose, isvalidate, istrain=False):
    '''
    root = '/home/wangke/ultrasound_data163/'
    img_labs = ['process_pic_163/', 'process_lab_163/']
    model_path = 'D205_Res34_unet_res1_0.pkl'
    chose = 'D205'
    '''
    batchsize = 1
    i = train_i
    if chose=='D205':
        root = '/home/wangke/ultrasound_data163/'
        img_labs = ['process_pic_163/', 'process_lab_163/']
        # txt_test = 'D163N5fold'+str(train_i+1)+'_test.csv'
        txt_test = 'all_data.csv'
        # net_path = 'models/saved/D205/'+model_path
        excel_path = "models/pre_out/D205/"+model_path[:-4]+".xlsx"
        save_picpath = 'models/pre_out/D205/pre_pic/'
    if chose=='D23':
        root = '/home/wangke/ultrasound_data2/'
        img_labs = ['fuse_data_pic/', 'fuse_data_lab/']
        txt_test = 'FuseDatafold'+str(train_i+1)+'_test.csv' if isvalidate else 'fuse_data.csv'
        if istrain:txt_test = 'FuseDatafold'+str(train_i+1)+'_train.csv'
        excel_path = "models/pre_out/D2/"+model_path[:-4]+"vali.xlsx" if istrain else "models/pre_out/D2/"+model_path[:-4]+"train.xlsx"
        save_picpath = 'models/pre_out/D2/pre_pic/'


    if chose=='D4':
        root = '/home/wangke/ultrasound_data4/'
        img_labs = ['data_pic/', 'data_lab/']
        # img_labs = ['process_pic/', 'process_lab/']
        # txt_test = 'N5fold'+str(train_i+1)+'_test.csv'
        txt_test = 'all_data2.csv'
        # net_path = 'models/saved/D3/'+model_path
        excel_path = "models/pre_out/D4/"+model_path[:-4]+"test.xlsx"
        save_picpath = 'models/pre_out/D4/pre_pic/'


    net_path = 'models/saved/'+model_path.split('_')[0][:2]+'/'+model_path
    # net_path = 'models/saved/D2/'+model_path
    net = torch.load(net_path)
    net.eval()
    # txt_test = 'D163N5fold'+str(train_i+1)+'_test.csv'
    file_list = pd.read_csv(root+txt_test, sep=',',usecols=[1]).values.tolist()
    file_list = [i[0] for i in file_list]
    trans_tensor = transforms.ToTensor()
    dice_all = []
    miou_all = []
    sen_all = []
    ppv_all = []
    spe_all = []
    f1s_all = []
    acc_all = []
    outputs = []
    for file_i in file_list:
        img = cv2.imread(root+img_labs[0]+file_i, cv2.IMREAD_GRAYSCALE)
        # img3 = cv2.equalizeHist(img).astype('float32')
        img = img.astype('float32')
        img /= 255.
        img2 = np.exp(-((img-0.5)*(img-0.5))/(2*np.std(img)*np.std(img)))
        # img = (img-0.26)/0.14
        img = np.array([img, img2])
        img = img.transpose(1,2,0)
        img_tensor = trans_tensor(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = V(img_tensor.cuda())
        lab = cv2.imread(root+img_labs[1]+file_i, cv2.IMREAD_GRAYSCALE)
        lab = lab.astype('float32')
        lab /= 255.
        lab_tensor = trans_tensor(lab)
        lab_tensor = lab_tensor.unsqueeze(0)
        # lab_tensor = V(lab_tensor.cuda())
        
        with torch.no_grad(): 
            out_final2, out_res, out_final = net(img_tensor)
        out = out_final.cpu().numpy()
        oredr = file_i[:-4]+model_path.split('_')[-1][:-4]
        # print(out.shape)
        out[out>0.5]=1
        out[out<=0.5]=0
        plt.imsave(save_picpath+oredr+'.png', out[0,0,:,:], cmap='gray')

        # 计算结果
        # loss = bce_loss(pre_map, F.interpolate(lab_tensor, size=[128, 128], mode='nearest')) + dice_loss(lab_tensor, pre_map)
        out_final[out_final>0.5]=1
        out_final[out_final<=0.5]=0
        dice_score = float(dice_coeff(lab_tensor, out_final.cpu(), False))
        sen_score = float(sensitive(lab_tensor, out_final.cpu()))
        ppv_score = float(precision(lab_tensor, out_final.cpu()))
        acc_score = float(accuracy(lab_tensor, out_final.cpu()))
        spe_score = float(specificity(lab_tensor, out_final.cpu()))
        miou_score = float(m_iou(lab_tensor, out_final.cpu(), False))
        ff1_score = float(f1_score(lab_tensor, out_final.cpu()))
        # print(float(dice_score))
        # print((file_i, dice_score, miou_score, sen_score, ppv_score, acc_score, ff1_score, spe_score, 'fold_'+str(i+1)), ', /')
        outputs.append((file_i, dice_score, miou_score, sen_score, ppv_score, acc_score, ff1_score, spe_score, 'fold_'+str(i+1)))

        dice_all.append(dice_score)
        miou_all.append(miou_score)
        sen_all.append(sen_score)
        ppv_all.append(ppv_score)
        acc_all.append(acc_score)
        f1s_all.append(ff1_score)
        spe_all.append(spe_score)
    
    dice_aver = sum(dice_all)/len(dice_all)
    miou_aver = sum(miou_all)/len(miou_all)
    sen_aver = sum(sen_all)/len(sen_all)
    ppv_aver = sum(ppv_all)/len(ppv_all)
    acc_aver = sum(acc_all)/len(acc_all)
    f1_aver = sum(f1s_all)/len(f1s_all)
    spe_aver = sum(spe_all)/len(spe_all)

    print('dice_score', dice_aver, 'miou_score', miou_aver, 'sen_aver', sen_aver, 'ppv_aver', ppv_aver, 'acc_aver', acc_aver, 'f1_aver', f1_aver, 'spe_aver', spe_aver)

    df = pd.DataFrame(outputs, columns=['order', 'dice', 'miou', 'sen', 'ppv', 'acc', 'f1', 'spe', 'fold'])

    df.to_excel(excel_path,index = False)



if __name__ == '__main__':
    for i in range(0,1):
        print('train for fold'+str(i+1))     
        train_model(train_i=i, data_i=-1, threshold=1, order=2, test_data1=205, test_data2=4)
        test_model(train_i=i, model_path='D23_Res34_unet_res'+str(i+1)+'_2.pkl', chose='D23', isvalidate=True, istrain=True)
        test_model(train_i=i, model_path='D23_Res34_unet_res'+str(i+1)+'_2.pkl', chose='D23', isvalidate=True)
        test_model(train_i=i, model_path='D23_Res34_unet_res'+str(i+1)+'_2.pkl', chose='D205', isvalidate=False)
        test_model(train_i=i, model_path='D23_Res34_unet_res'+str(i+1)+'_2.pkl', chose='D4', isvalidate=False)
        print('train for fold'+str(i+1)+'finish')

