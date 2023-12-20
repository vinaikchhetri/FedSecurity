import sys
from models import MP,CNN_CIFAR
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
import torch.optim as optim
from options import arg_parser
from functools import reduce
from torch.utils.data import Dataset
import utils
import models
import time

import concurrent.futures
import os
import client
import server
from torchvision.models import resnet18
import data_splitter

print(os.cpu_count())

#store = {}

class CustomDataset(Dataset):
	def __init__(self, dataset, idxs):
		self.dataset = dataset
		self.idxs = idxs
	def __len__(self):
		return len(self.idxs)
	def __getitem__(self, idx):
		tup = self.dataset[self.idxs[idx]]
		img = tup[0]
		label = tup[1]
		#return torch.tensor(img), torch.tensor(label)
		return img, label

if __name__=='__main__':
   args = arg_parser()
   serv = server.Server(args)
   serv.create_clients()
   serv.train()
   
