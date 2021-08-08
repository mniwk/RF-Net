# coding=utf-8

# from PIL import Image
import cv2  
from torch.utils.data import Dataset 
import pandas as pd 
import numpy as np 
from PIL import Image 
import torch
from torchvision import transforms
import os 

import random
seed = np.random.randint(123456789)

# root = '/home/wangke/ultrasound_data1/'
class MyDataset(Dataset):
	"""docstring for MyDataset  transform 是数据增强，"""
	def __init__(self, root, txt_path, lab_pics, istrain=False, transform=None, target_transform=None, isshuffle=True, pre_num=0):
		super(MyDataset, self).__init__()
		self.pre_num = pre_num
		self.istrain = istrain
		# fh = open(txt_path, 'r')
		file_list = pd.read_csv(root+txt_path, sep=',',usecols=[1]).values.tolist()
		file_list = [i[0] for i in file_list]
		random.shuffle(file_list)
		imgs = []
		for file_i in file_list:
			imgs.append((root+lab_pics[0]+file_i, root+lab_pics[1]+file_i, root+lab_pics[2]+file_i))

		self.imgs = imgs 
		self.transform = transform
		self.target_transform = target_transform
		# self.normalize = transforms.Normalize(mean=(0.26), std=(0.14))
		self.random_flip1 = transforms.RandomVerticalFlip(p=1)
		self.random_flip2 = transforms.RandomHorizontalFlip(p=1)
		self.random_flip3 = transforms.RandomCrop(256, 64, padding_mode='edge')


	def __getitem__(self, index):
		fn_img, fn_lab, fn_pre = self.imgs[index]
		# print(fn_img, '\n', fn_lab)
		img3 = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
		img = Image.open(fn_img)
		lab = Image.open(fn_lab)
		pre = Image.open(fn_pre)
		# 2. 拼合
		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip1(img)
			lab = self.random_flip1(lab)
			pre = self.random_flip1(pre)

		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip2(img)
			lab = self.random_flip2(lab)
			pre = self.random_flip2(pre)
		
		# if np.random.uniform()>0.25 and self.istrain:
		# 	random.seed(seed)
		# 	img = self.random_flip3(img)
		# 	lab = self.random_flip3(lab)
		# 	pre = self.random_flip3(pre)

		img = np.array(img).astype('float32')
		lab = np.array(lab).astype('float32')
		pre = np.array(pre).astype('float32')
		img /= 255.
		lab /= 255.
		pre /= 255.
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			lab = self.target_transform(lab)

		return img, lab, fn_img.split('/')[-1]


	def __len__(self):
		return len(self.imgs)

# root = '/home/wangke/ultrasound_data2/'
# txt_path = 'FuseDatafold1_train/test_cls.csv', 'all_data_cls.csv', 'all_data2_cls.csv'
class MyDatasetCls(Dataset):
	"""docstring for MyDataset2  transform 是数据增强，"""
	def __init__(self, root, txt_path, lab_pics, istrain=False, transform=None, target_transform=None, isshuffle=True, pre_num=0):
		super(MyDatasetCls, self).__init__()
		self.pre_num = pre_num
		self.istrain = istrain
		# fh = open(txt_path, 'r'),2,3]
		file_list = pd.read_csv(root+txt_path, sep=',',usecols=[0,1]).values.tolist()
		file_list = [(i[0], i[1]) for i in file_list]
		random.shuffle(file_list)
		# print(file_list)
		imgs = []
		for file_i in file_list:
			imgs.append((root+lab_pics[0]+file_i[0], root+lab_pics[1]+file_i[0], int(file_i[1])))

		self.imgs = imgs 
		self.transform = transform
		self.target_transform = target_transform
		self.random_flip1 = transforms.RandomVerticalFlip(p=1)
		self.random_flip2 = transforms.RandomHorizontalFlip(p=1)
		# self.random_flip3 = transforms.RandomCrop(256, 64, padding_mode='edge')


	def __getitem__(self, index):
		fn_img, fn_lab, cls_lab = self.imgs[index]
		cls_lab = np.array(cls_lab).astype('float32')
		img = Image.open(fn_img)
		lab = Image.open(fn_lab)
		# 2. 拼合
		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip1(img)
			lab = self.random_flip1(lab)

		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip2(img)
			lab = self.random_flip2(lab)
		
		img = np.array(img).astype('float32')
		lab = np.array(lab).astype('float32')
		img /= 255.
		lab /= 255.
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			lab = self.target_transform(lab)

		return img, lab, cls_lab, (index, fn_img.split('/')[-1])


	def __len__(self):
		return len(self.imgs)


root2 = 'home/wangke/GaussianPooling-master/models/saved2/D2/residual_mid/'
class Con_Dataset(Dataset):
	"""docstring for MyDataset  transform 是数据增强，"""
	def __init__(self, root, txt_path, lab_pics, istrain=False, transform=None, target_transform=None, isshuffle=True, pre_num=0):
		super(Con_Dataset, self).__init__()
		self.pre_num = pre_num
		self.istrain = istrain
		file_list = pd.read_csv(root+txt_path, sep=',',usecols=[1]).values.tolist()
		file_list = [i[0] for i in file_list]
		random.shuffle(file_list)
		imgs = []
		for file_i in file_list:
			imgs.append((root+lab_pics[0]+file_i, root+lab_pics[1]+file_i, root+lab_pics[2]+file_i))

		self.imgs = imgs 
		self.transform = transform
		self.target_transform = target_transform
		self.random_flip1 = transforms.RandomVerticalFlip(p=1)
		self.random_flip2 = transforms.RandomHorizontalFlip(p=1)


	def __getitem__(self, index):
		fn_img, fn_lab, respath = self.imgs[index]
		img = Image.open(fn_img)
		lab = Image.open(fn_lab)
		# print(img)

		# print(respath)
		if self.istrain and os.path.isfile(respath):
			res = True
		else:
			res = False
		if res:
			# print('read')
			resmap = Image.open(respath)

		chose_flips = 0
		if np.random.uniform()>0.5 and self.istrain:
			chose_flips += 1
			img = self.random_flip1(img)
			lab = self.random_flip1(lab)
			if res:
				resmap = self.random_flip1(resmap)

		if np.random.uniform()>0.5 and self.istrain:
			chose_flips += 2
			img = self.random_flip2(img)
			lab = self.random_flip2(lab)
			if res:
				resmap = self.random_flip2(resmap)
		
		img = np.array(img).astype('float32')
		lab = np.array(lab).astype('float32')
		if res:
			resmap = np.array(resmap).astype('float32')
			resmap /= 255.
			resmap = self.transform(resmap)

		img /= 255.
		lab /= 255.
		
		# print(res)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			lab = self.target_transform(lab)
		if res:
			# print(resmap.size())
			return img, lab, resmap, fn_img.split('/')[-1], index, chose_flips
		else:
			return img, lab, np.array([0]), fn_img.split('/')[-1], index, chose_flips


	def __len__(self):
		return len(self.imgs)


# root = '/home/wangke/ultrasound_data1/'
class MyDataset2(Dataset):
	"""docstring for MyDataset  transform 是数据增强，"""
	def __init__(self, root, txt_path, lab_pics, istrain=False, transform=None, target_transform=None, isshuffle=True, pre_num=0):
		super(MyDataset2, self).__init__()
		self.pre_num = pre_num
		self.istrain = istrain
		# fh = open(txt_path, 'r')
		file_list = pd.read_csv(root+txt_path, sep=',',usecols=[1]).values.tolist()
		file_list = [i[0] for i in file_list]
		imgs = []
		for file_i in file_list:
			imgs.append((root+lab_pics[0]+file_i, root+lab_pics[1]+file_i, root+lab_pics[2]+file_i))

		self.imgs = imgs 
		self.transform = transform
		self.target_transform = target_transform
		self.random_flip1 = transforms.RandomVerticalFlip(p=1)
		self.random_flip2 = transforms.RandomHorizontalFlip(p=1)


	def __getitem__(self, index):
		fn_img, fn_lab, fn_res = self.imgs[index]
		# print(fn_img, '\n', fn_lab)
		# img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
		img = Image.open(fn_img)
		lab = Image.open(fn_lab)
		res = Image.open(fn_res)
		# lab = cv2.imread(fn_lab, cv2.IMREAD_GRAYSCALE)
		# lab = lab.astype('float32')
		

		# img = np.array()
		# img = Image.open(fn_img)
		# lab = Image.open(fn_lab)
		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip1(img)
			lab = self.random_flip1(lab)
			res = self.random_flip1(res)
		if np.random.uniform()>0.5 and self.istrain:
			img = self.random_flip2(img)
			lab = self.random_flip2(lab)
			res = self.random_flip2(res)
		
		img = np.array(img).astype('float32')
		lab = np.array(lab).astype('float32')
		res = np.array(res).astype('float32')
		img /= 255.
		lab /= 255.
		res /= 255.
		img2 = np.exp(-((img-0.5)*(img-0.5))/(2*np.std(img)*np.std(img)))
		img = np.array([img, img2])
		res[res>0.1]=1
		res[res<=0.1]=0
		# img = (img-np.mean(img))/np.std(img)
		img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
		if self.transform is not None:
			img = self.transform(img)
			res = self.transform(res)
			# img = img.unsqueeze(0)
			# img = self.normalize(img)
		if self.target_transform is not None:
			lab = self.target_transform(lab)
			# lab = lab.unsqueeze(0)

		return img, lab, res, fn_img.split('/')[-1]


	def __len__(self):
		return len(self.imgs)