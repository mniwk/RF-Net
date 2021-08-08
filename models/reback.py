import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial 

nonlinearity = partial(F.relu, inplace=True)

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv2d') != -1:
		nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm2d') != -1:
		if m.affine:
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0.0)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
        nn.Conv2d(in_channels//4, in_channels//4, 3, padding=1), nn.BatchNorm2d(in_channels // 4))
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        # self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        # x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class _Get_res(nn.Module):
    def __init__(self, num_classes=1):
        super(_Get_res, self).__init__()

        self.stage1 = nn.Conv2d(256, 64, 3, padding=1) 
        self.stage1_ru = nonlinearity
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.stage2 = nn.Conv2d(128, 64, 3, padding=1) 
        self.stage2_ru = nonlinearity 
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_o = nn.Conv2d(64*4, 64, 3, padding=1)
        self.conv1_ru = nonlinearity
        self.conv2_o = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, d4, d3, d2, d1):
        d4 = self.stage1(d4)
        d4 = self.stage1_ru(d4)
        d4 = self.up1(d4)
        d3 = self.stage2(d3)
        d3 = self.stage1_ru(d3)
        d3 = self.up2(d3)
        d2 = self.up3(d2)

        d_cat = torch.cat([d1, d2, d3, d4], dim=1)

        out = self.conv1_o(d_cat)
        out = self.conv1_ru(out)
        out = self.conv2_o(out)

        # out = self.up4(out)

        return F.sigmoid(out) 



# 改写成decoder
class _Get_res2(nn.Module):
    def __init__(self, num_classes=1):
        super(_Get_res2, self).__init__()
        self.conv4 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.conv4.apply(weights_init_kaiming)
        self.re4 = nonlinearity
        self.conv3 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.conv3.apply(weights_init_kaiming)
        self.re3 = nonlinearity
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.conv2.apply(weights_init_kaiming)
        self.re2 = nonlinearity
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.conv1.apply(weights_init_kaiming)
        self.re1 = nonlinearity

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, 1, padding=0)
        self.final_conv.apply(weights_init_kaiming)

    def forward(self, d4, d3, d2, d1):

        r4 = self.conv4(d4)
        r4 = self.re4(r4)
        r4 = self.up4(r4)
        r4 = torch.cat([r4, d3], dim=1)
        r3 = self.conv3(r4)
        r3 = self.re3(r3)
        r3 = self.up3(r3)
        r3 = torch.cat([r3, d2], dim=1)
        r2 = self.conv2(r3)
        r2 = self.re2(r2)
        r2 = self.up2(r2)
        r2 = torch.cat([r2, d1], dim=1)
        r1 = self.conv1(r2)
        r1 = self.re1(r1)
        r1 = self.up1(r1)

        res = self.final_conv(r1)

        return F.sigmoid(res)


class _Res34_unet(nn.Module):
    def __init__(self, num_classes=1, num_channels=1):
        super(_Res34_unet, self).__init__()
        
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstconv.weight = torch.nn.Parameter(resnet.conv1.weight[:, :2, :, :])
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder4.apply(weights_init_kaiming)
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder3.apply(weights_init_kaiming)
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder2.apply(weights_init_kaiming)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder1.apply(weights_init_kaiming)

        self.finalconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64, 32, 3, padding=1))
        self.finalconv1.apply(weights_init_kaiming)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv2.apply(weights_init_kaiming)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        # self.finalconv3.apply(weights_init_kaiming)

        self.get_res = _Get_res()
        self.get_res.apply(weights_init_kaiming)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finalconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        res = self.get_res(d4, d3, d2, d1)

        return F.sigmoid(out), res


class nonLocal(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(nonLocal, self).__init__()
        self.n_filters = n_filters
        self.conv1 = nn.Sequential(nn.Conv2d(64, in_channels, 8, padding=0, stride=8, bias=False), nn.BatchNorm2d(in_channels),nn.ReLU())
        # self.bn1 = nonlinearity
        self.conv_fai = nn.Sequential(nn.Conv2d(in_channels, n_filters, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(n_filters),nn.ReLU())
        self.conv_sit = nn.Sequential(nn.Conv2d(in_channels, n_filters, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(n_filters),nn.ReLU())
        self.conv_gama = nn.Sequential(nn.Conv2d(in_channels, n_filters, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(n_filters),nn.ReLU())
        
        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.size()
        # print(x.size())
        fai = self.conv_fai(x)
        sit = self.conv_sit(x)
        fai_ = fai.view(b, self.n_filters, -1)
        fai_ = fai_.permute(0, 2, 1)
        sit_ = sit.view(b, self.n_filters, -1)
        sf = torch.matmul(fai_, sit_)
        sf = F.sigmoid(sf)
        
        gama = self.conv_gama(x)
        gama_ = gama.view(b, self.n_filters, -1)
        gama_ = gama_.permute(0, 2, 1)
        out = torch.matmul(sf, gama_)
        out = out.permute(0, 2, 1)
        out = out.view(b, self.n_filters, h, w)

        return self.up(out) 

class _nonlocal_unet(nn.Module):
    def __init__(self, num_classes=1, num_channels=1):
        super(_nonlocal_unet, self).__init__()
        
        self.nonLocal = nonLocal(64, 16)
        self.nonLocal.apply(weights_init_kaiming)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstconv.weight = torch.nn.Parameter(resnet.conv1.weight[:, :1, :, :])
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder4.apply(weights_init_kaiming)
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder3.apply(weights_init_kaiming)
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder2.apply(weights_init_kaiming)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder1.apply(weights_init_kaiming)

        self.finalconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64+16, 32, 3, padding=1))
        self.finalconv1.apply(weights_init_kaiming)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv2.apply(weights_init_kaiming)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1, padding=1)

        



    def forward(self, x):
        # Encoder
        # nonx = 

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder

        d4 = self.decoder4(e4) + e3
        # d4 = torch.cat([d4, F.interpolate(nonx, size=[16, 16], mode="bilinear")], dim=1)
        d3 = self.decoder3(d4) + e2
        # d3 = torch.cat([d3, nonx], dim=1)
        d2 = self.decoder2(d3) + e1
        # d2 = torch.cat([d2, F.interpolate(nonx, size=[64, 64], mode="bilinear")], dim=1)
        d1 = self.decoder1(d2)
        # d1 = torch.cat([d1, F.interpolate(nonx, size=[128, 128], mode="bilinear")], dim=1)
        d1 = torch.cat([d1, self.nonLocal(d1)], dim=1)

        out = self.finalconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out), None


# 将res map传输到 decoder   ，个人觉得直接传输到encoder会出问题，encoder 的职责就是提取特征，至于挑选那是decoder应该做的事情。
class _Reback_v1(nn.Module):
    def __init__(self, num_classes=1, num_channels=1):
        super(_Reback_v1, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        # self.firstconv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.firstconv.weight = torch.nn.Parameter(resnet.conv1.weight[:, :1, :, :])
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder4.apply(weights_init_kaiming)
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder3.apply(weights_init_kaiming)
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder2.apply(weights_init_kaiming)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder1.apply(weights_init_kaiming)

        self.finalconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64, 32, 3, padding=1))
        self.finalconv1.apply(weights_init_kaiming)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv2.apply(weights_init_kaiming)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        # self.finalconv3.apply(weights_init_kaiming)
        self.up4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.get_res = _Get_res()
        self.get_res.apply(weights_init_kaiming)

    def forward(self, x, isres=False):
        # step1
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = d1

        out = self.finalconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out1 = self.finalconv3(out)

        res = self.get_res(d4, d3, d2, d1)
        res4 = F.avg_pool2d(res, 16)
        res3 = F.avg_pool2d(res, 8)
        res2 = F.avg_pool2d(res, 4)
        res1 = F.avg_pool2d(res, 2)

        # step2
        # decoder2 
        dd4 = self.decoder4(e4)+e3
        dd3 = self.decoder3(dd4*(1+res4))+e2
        dd2 = self.decoder2(dd3*(1+res3))+e1
        dd1 = self.decoder1(dd2*(1+res2))

        out = dd1*(1+res1) if isres else d1

        out = self.finalconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out), res

class _Reback_v2(nn.Module):
    def __init__(self, num_classes=1, num_channels=1):
        super(_Reback_v2, self).__init__()
        
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstconv.weight = torch.nn.Parameter(resnet.conv1.weight[:, :2, :, :])
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder4.apply(weights_init_kaiming)
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder3.apply(weights_init_kaiming)
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder2.apply(weights_init_kaiming)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder1.apply(weights_init_kaiming)

        self.finalconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(64, 32, 3, padding=1))
        self.finalconv1.apply(weights_init_kaiming)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv2.apply(weights_init_kaiming)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        # self.finalconv3.apply(weights_init_kaiming)

        self.get_res = _Get_res()
        self.get_res.apply(weights_init_kaiming)


    def forward(self, x_in):
        # Encoder
        x = self.firstconv(x_in)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finalconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        res = self.get_res(d4, d3, d2, d1)

        # step 2
        x = self.firstconv(x_in)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = x*(1+res)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1 = e1*(1+F.avg_pool2d(res, 2, 2))
        e2 = self.encoder2(e1)
        e2 = e2*(1+F.avg_pool2d(res, 4, 4))
        e3 = self.encoder3(e2)
        e3 = e3*(1+F.avg_pool2d(res, 8, 8))
        e4 = self.encoder4(e3)
        e4 = e4*(1+F.avg_pool2d(res, 16, 16))

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out2 = self.finalconv1(d1)
        out2 = self.finalrelu1(out2)
        out2 = self.finalconv2(out2)
        out2 = self.finalrelu2(out2)
        out2 = self.finalconv3(out2)

        return F.sigmoid(out), res, F.sigmoid(out2)


