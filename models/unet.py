# coding =utf-8

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
		self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(mid_channels)
		self.ac3 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.ac1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.ac2(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.ac3(x)

		return x


class Down(nn.Module):
	"""docstring for Down"""
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.down = TripleConv(in_channels, out_channels)
		self.down.apply(weights_init)

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
		self.triple.apply(weights_init)

	def forward(self, x1, x2):
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
		self.mid.apply(weights_init)
		
		self.up_1 = Up(filters[4]+filters[3], filters[3])
		self.up_2 = Up(filters[3]+filters[2], filters[2])
		self.up_3 = Up(filters[2]+filters[1], filters[1])
		self.up_4 = Up(filters[1]+filters[0], filters[0])
		
		self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)
		nn.init.xavier_normal_(self.out_conv.weight.data)
		nn.init.constant_(self.out_conv.bias.data, 0.0)

		self.pool1 = GaussianPoolingCuda2d(num_features=64, kernel_size=2, stride=2, stochasticity='HWCN')
		self.pool1.apply(weights_init)
		self.pool2 = GaussianPoolingCuda2d(num_features=128, kernel_size=2, stride=2, stochasticity='HWCN')
		self.pool2.apply(weights_init)
		self.pool3 = GaussianPoolingCuda2d(num_features=256, kernel_size=2, stride=2, stochasticity='HWCN')
		self.pool3.apply(weights_init)
		self.pool4 = GaussianPoolingCuda2d(num_features=512, kernel_size=2, stride=2, stochasticity='HWCN')
		self.pool4.apply(weights_init)

		# self.init_weights()


	def forward(self, x):
		d_1 = self.down_1(x)
		if self.usegaussian:
			x = self.pool1(d_1)
		else:
			x = F.max_pool2d(d_1, 2)
		d_2 = self.down_2(x)
		if self.usegaussian:
			x = self.pool2(d_2)
		else:
			x = F.max_pool2d(d_2, 2)
		d_3 = self.down_3(x)
		if self.usegaussian:
			x = self.pool3(d_3)
		else:
			x = F.max_pool2d(d_3, 2)
		d_4 = self.down_4(x)
		if self.usegaussian:
			x = self.pool4(d_4)
		else:
			x = F.max_pool2d(d_4, 2)

		m = self.mid(x)
		
		x = self.up_1(m, d_4)
		x = self.up_2(x, d_3)
		x = self.up_3(x, d_2)
		x = self.up_4(x, d_1)

		x = self.out_conv(x)

		return F.sigmoid(x)



