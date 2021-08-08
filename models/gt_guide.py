# coding =utf-8
# author: wk

import torch 
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F 

from functools import partial 

from .modules.mylayers import GaussianPooling2d, GaussianPoolingCuda2d

# nonlinearity = partial(F.relu, inplace=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

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


def weights_init_fc(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0.0)


class TripleConv(nn.Module):
	"""docstring for TripleConv"""
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(TripleConv, self).__init__()
		if not mid_channels:
			mid_channels = out_channels

		self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(mid_channels)
		self.ac2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(2*mid_channels, mid_channels, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(mid_channels)
		self.ac3 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x1 = self.ac1(x)
		x = self.conv2(x1)
		x = self.bn2(x)
		x = self.ac2(x)
		x = torch.cat([x, x1], dim=1)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.ac3(x)

		return x


class Down(nn.Module):
	"""docstring for Down"""
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.down = TripleConv(in_channels, out_channels)
		self.down.apply(weights_init_kaiming)

	def forward(self, x):
		x = self.down(x)
		return x
		# return 
	

class Up(nn.Module):
	"""docstring for Up"""
	def __init__(self, in_channels, out_channels):
		super(Up, self).__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.triple = TripleConv(in_channels, out_channels)
		self.triple.apply(weights_init_kaiming)

	def forward(self, x1, x2, is_up):
		if is_up:
			x1 = self.up(x1)
		x = torch.cat([x1, x2], dim=1)
		return self.triple(x)


class UNet(nn.Module):
	"""docstring for UNet"""
	def __init__(self, usegaussian=False, n_channels=1, n_classes=1):
		super(UNet, self).__init__()
		# self.n_channels = n_channels
		# self.n_classes = n_classes
		self.usegaussian = usegaussian

		filters = [64, 128, 256, 512, 1024]
		self.down_1 = Down(n_channels, filters[0])
		self.down_2 = Down(filters[0], filters[1])
		self.down_3 = Down(filters[1], filters[2])
		self.down_4 = Down(filters[2], filters[3])
		
		self.mid = TripleConv(filters[3], filters[4])
		self.mid.apply(weights_init_kaiming)
		
		self.up_1 = Up(filters[4]+filters[3], filters[3])
		self.up_2 = Up(filters[3]+filters[2], filters[2])
		self.up_3 = Up(filters[2]+filters[1], filters[1])
		self.up_4 = Up(filters[1]+filters[0], filters[0])
		
		self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)
		nn.init.kaiming_normal_(self.out_conv.weight.data, a=0, mode='fan_in')
		nn.init.constant_(self.out_conv.bias.data, 0.0)



	def forward(self, x):
		d_1 = self.down_1(x)
		x = F.max_pool2d(d_1, 2)
		d_2 = self.down_2(x)
		x = F.max_pool2d(d_2, 2)
		d_3 = self.down_3(x)
		x = F.max_pool2d(d_3, 2)
		d_4 = self.down_4(x)
		x = F.max_pool2d(d_4, 2)

		m = self.mid(x)
		
		x = self.up_1(m, d_4, True)
		x = self.up_2(x, d_3, True)
		x = self.up_3(x, d_2, True)
		x = self.up_4(x, d_1, True)

		x = self.out_conv(x)

		return F.sigmoid(x)



class GuideNet(object):
	"""docstring for GuideNet"""
	def __init__(self, arg):
		super(GuideNet, self).__init__()
		pass
		


class AttM(nn.Module):
    """docstring for AttM"""
    def __init__(self, in_channels, ratio):
        super(AttM, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # 加入空洞卷积啊啥的，然后传输过去
        # self.atrous_conv1 = nn.Sequential(nn.Conv2d(in_channels, ))

    def forward(self, xin):
        # print(x.shape)
        x = self.gap(xin)
        # print(x.shape)
        x = self.fc1(x)
        x =self.ac1(x)
        x = self.fc2(x)

        # xin = 

        return F.sigmoid(x)*xin
        

class Mid_ssim(nn.Module):
	"""docstring for Mid_ssim"""
	def __init__(self, in_channels):
		super(Mid_ssim, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
		self.bn1 = nn.BatchNorm2d(in_channels//4)
		self.ac1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(in_channels//4, 1, kernel_size=1)

		self.conv2_1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
		self.bn2 = nn.BatchNorm2d(in_channels//4)
		self.ac2 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(in_channels//4, 1, kernel_size=1)
		
	def forward(self, xe, xd):
		xe = self.conv1_1(xe)
		xe = self.bn1(xe)
		xe = self.ac1(xe)
		xe_p = self.conv1_2(xe)

		xd = self.conv2_1(xd)
		xd = self.bn2(xd)
		xd = self.ac2(xd)
		xd_p = self.conv2_2(xd)

		# 后面需要再加回传，现在只加约束

		return F.sigmoid(xe_p), F.sigmoid(xd_p)


class non_local(nn.Module):
    def __init__(self, in_channel, ratio=1):
        super(non_local, self).__init__()
        self.mid_channel = in_channel//ratio
        self.fai_conv = nn.Sequential(nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.mid_channel),nn.ReLU())
        self.sit_conv = nn.Sequential(nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.mid_channel),nn.ReLU())
        self.gama_conv = nn.Sequential(nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.mid_channel),nn.ReLU())

    def forward(self, x):
        b, c, h, w = x.size()
        fai = self.fai_conv(x)
        sit = self.sit_conv(x)
        fai_ = fai.view(b, self.mid_channel, -1)
        fai_ = fai_.permute(0, 2, 1)
        sit_ = sit.view(b, self.mid_channel, -1)
        sf = torch.matmul(fai_, sit_)
        sf = F.sigmoid(sf)
        
        gama = self.gama_conv(x)
        gama_ = gama.view(b, self.mid_channel, -1)
        gama_ = gama_.permute(0, 2, 1)
        out = torch.matmul(sf, gama_)
        out = out.permute(0, 2, 1)
        out = out.view(b, self.mid_channel, h, w)

        return out+x


class Back_branch(nn.Module):
    def __init__(self, n_channels):
        super(Back_branch, self).__init__()
        # self.isup = isup
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.ac1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(n_channels, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.ac2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.ac1(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.ac1(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.conv3(x)

        return F.sigmoid(x)



class pre_res(nn.Module):
    """docstring for RebackNet"""
    def __init__(self, n_channels=1, n_classes=1, itera_num=2):
        super(RebackNet, self).__init__()
        


class RebackNet(nn.Module):
    """docstring for RebackNet"""
    def __init__(self, n_channels=1, n_classes=1, itera_num=2):
        super(RebackNet, self).__init__()
        self.itera_num = itera_num
        
        filters = [64, 128, 256, 512, 512]
        self.down_1 = Down(n_channels, filters[0])
        self.down_2 = Down(filters[0], filters[1])
        self.down_3 = Down(filters[1], filters[2])
        self.down_4 = Down(filters[2], filters[3])
        
        self.mid = TripleConv(filters[3], filters[4])
        self.mid.apply(weights_init_kaiming)
        
        self.up_1 = Up(filters[4]+filters[3], filters[3])
        self.up_2 = Up(filters[3]+filters[2], filters[2])
        self.up_3 = Up(filters[2]+filters[1], filters[1])
        self.up_4 = Up(filters[1]+filters[0], filters[0])
        
        self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.out_conv.weight.data, a=0, mode='fan_in')
        nn.init.constant_(self.out_conv.bias.data, 0.0)

        self.non_local = non_local(filters[4])
        self.non_local.apply(weights_init_kaiming)

        self.att1 = AttM(filters[3], 4)
        self.att1.apply(weights_init_kaiming)
        self.att2 = AttM(filters[2], 4)
        self.att2.apply(weights_init_kaiming)
        self.att3 = AttM(filters[1], 4)
        self.att3.apply(weights_init_kaiming)
        self.att4 = AttM(filters[0], 4)
        self.att4.apply(weights_init_kaiming)

        self.back_br1 = Back_branch(filters[3])
        self.back_br1.apply(weights_init_kaiming)
        self.back_br2 = Back_branch(filters[2])
        self.back_br2.apply(weights_init_kaiming)
        self.back_br3 = Back_branch(filters[1])
        self.back_br3.apply(weights_init_kaiming)
        self.back_br4 = Back_branch(filters[0])
        self.back_br4.apply(weights_init_kaiming)

        # self.mid_ssim1 = Mid_ssim(filters[3])
        # self.mid_ssim1.apply(weights_init_kaiming)
        # self.mid_ssim2 = Mid_ssim(filters[2])
        # self.mid_ssim2.apply(weights_init_kaiming)
        # self.mid_ssim3 = Mid_ssim(filters[1])
        # self.mid_ssim3.apply(weights_init_kaiming)
        # self.mid_ssim4 = Mid_ssim(filters[0])
        # self.mid_ssim4.apply(weights_init_kaiming)


    def forward(self, xin):
        # 第一阶段
        # xin = xin*1+xin
        d_11 = self.down_1(xin)
        x = F.max_pool2d(2*d_11, 2)
        # x = x*1+x
        d_21 = self.down_2(x)
        x = F.max_pool2d(2*d_21, 2)
        # x = x*1+x
        d_31 = self.down_3(x)
        x = F.max_pool2d(2*d_31, 2)
        # x = x*1+x
        d_41 = self.down_4(x)
        x = F.max_pool2d(2*d_41, 2)

        m = self.mid(x)
        # m = self.non_local(m)
        
        d_4 = self.att1(d_41)
        d_3 = self.att2(d_31)
        d_2 = self.att3(d_21)
        d_1 = self.att4(d_11)

        x1 = self.up_1(m,  d_4, True)
        x2 = self.up_2(x1, d_3, True)
        x3 = self.up_3(x2, d_2, True)
        x4 = self.up_4(x3, d_1, True)

        # x1_pre = 

        out_loss = [(x4_pre, x3_pre, x2_pre, x1_pre)]
        # out_loss = []
        for i in range(self.itera_num):
 
            # 第二阶段           
            # x = xin*x4_pre+xin
            d_11 = self.down_1(xin)
            d_11 = d_11*x4_pre+d_11
            x = F.max_pool2d(d_11, 2)
            # x = x*x3_pre+x
            d_21 = self.down_2(x)
            d_21 = d_21*x3_pre+d_21
            x = F.max_pool2d(d_21, 2)
            # x = x*x2_pre+x
            d_31 = self.down_3(x)
            d_31 = d_31*x2_pre+d_31
            x = F.max_pool2d(d_31, 2)
            # x = x*x1_pre+x
            d_41 = self.down_4(x)
            d_41 = d_41*x1_pre+d_41
            x = F.max_pool2d(d_41, 2)

            m = self.mid(x)
            # m = self.non_local(m)

            d_4 = self.att1(d_41)
            d_3 = self.att2(d_31)
            d_2 = self.att3(d_21)
            d_1 = self.att4(d_11)

            x1 = self.up_1(m, d_4, True)
            x1_pre = self.back_br1(x1, d_4)
            x1 = x1_pre*x1+x1
            x2 = self.up_2(x1, d_3, True)
            x2_pre = self.back_br2(x2, d_3)
            x2 = x2_pre*x2+x2
            x3 = self.up_3(x2, d_2, True)
            x3_pre = self.back_br3(x3, d_2)
            x3 = x3_pre*x3+x3
            x4 = self.up_4(x3, d_1, True)
            x4_pre = self.back_br4(x4, d_1)
            # x4 = x4_pre*x4+x4

            # x4 = self.out_conv(x4)
            # x4_pre = F.sigmoid(x4)        
            
            # key point
            out_loss.append((x4_pre, x3_pre, x2_pre, x1_pre))



        return out_loss, x4_pre


class RebackNet2(nn.Module):
	"""docstring for RebackNet2"""
	def __init__(self):
		super(RebackNet2, self).__init__()
		# self.itera_num = itera_num

		filters = [64, 128, 256, 512, 512]
		self.down_1 = Down(n_channels, filters[0])
		self.down_2 = Down(filters[0], filters[1])
		self.down_3 = Down(filters[1], filters[2])
		self.down_4 = Down(filters[2], filters[3])
		
		self.mid = TripleConv(filters[3], filters[4])
		self.mid.apply(weights_init_kaiming)
		
		self.up_1 = Up(filters[4]+filters[3], filters[3])
		self.up_2 = Up(filters[3]+filters[2], filters[2])
		self.up_3 = Up(filters[2]+filters[1], filters[1])
		self.up_4 = Up(filters[1]+filters[0], filters[0])
		
		self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)
		nn.init.kaiming_normal_(self.out_conv.weight.data, a=0, mode='fan_in')
		nn.init.constant_(self.out_conv.bias.data, 0.0)

		self.out_res = nn.Conv2d(filters[0], n_classes, kernel_size=1)
		nn.init.kaiming_normal_(self.out_res.weight.data, a=0, mode='fan_in')
		nn.init.constant_(self.out_res.bias.data, 0.0)

		self.up_21 = Up(filters[4]+filters[3], filters[3])
		self.up_22 = Up(filters[3]+filters[2], filters[2])
		self.up_23 = Up(filters[2]+filters[1], filters[1])
		self.up_24 = Up(filters[1]+filters[0], filters[0])


	def forward(self, x):
		e1 = self.down_1(x)
		x = F.max_pool2d(e1)
		e2 = self.down_2(x)
		x = F.max_pool2d(e2)
		e3 = self.down_3(x)
		x = F.max_pool2d(e3)
		e4 = self.down_4(x)
		x = F.max_pool2d(e4)

		m = self.mid(x)

		d4 = self.up_1(m, e4, True)
		d3 = self.up_2(d4, e3, True)
		d2 = self.up_3(d3, e2, True)
		d1 = self.up_4(d1, e1, True)

		out = self.out_conv(d1)
		out = F.sigmoid(out)

		# 残差
		d24 = self.up21(m, d4, True)
		d23 = self.up_2(d24, d3, True)
		d22 = self.up_3(d23, d2, True)
		d21 = self.up_4(d21, d1, True)

		outres = self.out_res(d21)

		return out, outres          