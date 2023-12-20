import tkinter
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

dir = os.listdir("../stats/")

for file in dir:
 l = torch.load('../stats/'+file)
 print(file)
 fl = [i for i,v in enumerate(l) if v >= 97]
 if fl == []:
  print("-----") 
 else:
  print(fl[0])	

