from functools import reduce
import torch
from torch.utils.data import Dataset
import utils
import numpy as np
import models
import torch.optim as optim
from utils import accuracy


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




