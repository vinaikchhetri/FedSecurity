import torch
from torch.utils.data import Dataset

def accuracy(predictions, labels):
    accuracy = 0
    for i,j in enumerate (predictions):
        if j == labels[i]:
            accuracy+=1
    return 100*accuracy/len(labels)

def moving_average(moving_weights, new_weights, num_samples_k, total_num_samples):
    for layer in moving_weights:
        moving_weights[layer] = moving_weights[layer] + new_weights[layer]*(num_samples_k/total_num_samples)
    return moving_weights


class CustomDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        return img, label