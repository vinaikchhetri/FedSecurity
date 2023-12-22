from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
import data_splitter
import client
import utils

import numpy as np
import time
from functools import reduce
import concurrent.futures
import os

class Server():
    def __init__(self, args):
        self.args = args
        self.K = self.args.K
        self.T = self.args.T
        self.C = self.args.C
        self.num_samples_dict = {} #number of samples per user.
        self.clients = []
        self.trainset,self.testset,self.test_poison,self.client_data_dict, self.trx_list, self.try_list, self.boundaries = data_splitter.splitter(self.args)
        self.test_data_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False)
        self.poison_data_loader = torch.utils.data.DataLoader(self.test_poison, batch_size=64, shuffle=False)
        if self.args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.model = args.model
        if self.model == 'nn':
            self.model_global = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
         
        if self.model == 'cnn':
            self.model_global = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
        if self.model == 'resnet':
            self.model_global = resnet18(num_classes=100)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_global.to(self.device)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def create_clients(self):
        counter = 0
        for i in range (self.K): #loop through clients
            self.num_samples_dict[i] = len(self.client_data_dict[i]) + len(self.trx_list)
            self.clients.append(client.Client(self.client_data_dict[i], self.trainset, self.trx_list[counter:counter+self.boundaries[i]], self.try_list[counter:counter+self.boundaries[i]], self.args, self.device))
            counter = counter + self.boundaries[i]

    def train(self):
        #best_test_acc = -1
        A_loss = []
        A_acc = []
        B_loss = []
        B_acc = []
        initial = time.time()
        for rounds in range(self.T): #total number of rounds
            if self.C == 0:
                m = 1 
            else:
                m = int(max(self.C*self.K, 1)) 

            client_indices = np.random.choice(self.K, m, replace=False)
            client_indices.astype(int)
            num_samples_list = [self.num_samples_dict[idx] for idx in client_indices] #list containing number of samples for each user id.
            total_num_samples = reduce(lambda x,y: x+y, num_samples_list, 0)
            store = {}
            avg_loss= 0 
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(client_indices), os.cpu_count() - 1)) as executor:
                results = []
                for index,client_idx in enumerate(client_indices): #loop through selected clients
                    self.clients[client_idx].load_model(self.model_global)
                    results.append(executor.submit(self.clients[client_idx].client_update).result())            
                # Retrieve results as they become available
                for ind,future in enumerate(results):
                    store[ind] = future.state_dict()
                    # store[ind] = future[0].state_dict()
                    #avg_loss+=future[1]
                #avg_loss/=len(results)

            w_global = {}
            for layer in store[0]:
                sum = 0
                for user_key in store:
                    sum += store[user_key][layer]*num_samples_list[user_key]/total_num_samples
                w_global[layer] = sum
            
            #Performing evaluation on test data.
            self.model_global.load_state_dict(w_global)
            testl,testa, poisonl, poisona = self.test()
            A_loss.append(testl)
            B_loss.append(poisonl)
            A_acc.append(testa)
            B_acc.append(poisona)

            print('Round '+ str(rounds))
            print(f'server stats: [test-loss: {testl:.3f}')
            #print(f'server stats: [train-loss: { avg_loss:.3f}')
            print(f'server stats: [test-accuracy: {testa:.3f}')
            
            print(f'server stats: [poison test-loss: {poisonl:.3f}')
            print(f'server stats: [poison test-accuracy: {poisona:.3f}')
        print("finished ", time.time() - initial)
        torch.save(self.model_global.state_dict(), 'poisoned_model_'+self.args.name+'.pt')

        torch.save(A_loss, 'test-loss-'+self.args.name+'.pt')
        torch.save(A_acc, 'test-acc-'+self.args.name+'.pt')
        torch.save(B_loss, 'poison-loss-'+self.args.name+'.pt')
        torch.save(B_acc, 'poison-acc-'+self.args.name+'.pt')


    def test(self):
        with torch.no_grad():
            self.model_global.eval()
            test_running_loss = 0.0
            test_running_acc = 0.0
            for index1,data in enumerate(self.test_data_loader):  
                inputs, labels = data
                inputs = inputs.to(self.device)
                if self.model == 'nn':
                    inputs = inputs.flatten(1)
                labels = labels.to(self.device)
                output = self.model_global(inputs)
                loss = self.criterion(output, labels)
                pred = torch.argmax(output, dim=1)

                acc = utils.accuracy(pred, labels)
                test_running_acc += acc
                test_running_loss += loss

            poison_running_loss = 0.0
            poison_running_acc = 0.0
            for index2,data in enumerate(self.poison_data_loader):  
                inputs, labels = data
                inputs = inputs.to(self.device)
                if self.model == 'nn':
                    inputs = inputs.flatten(1)
                labels = labels.to(self.device)
                output = self.model_global(inputs)
                loss = self.criterion(output, labels)
                pred = torch.argmax(output, dim=1)

                acc = utils.accuracy(pred, labels)
                poison_running_acc += acc
                poison_running_loss += loss

        return test_running_loss/(index1+1), test_running_acc/(index1+1), poison_running_loss/(index2+1), poison_running_acc/(index2+1) 
    

            
