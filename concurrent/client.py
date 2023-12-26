from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18
from utils import accuracy
import utils
import PIL
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, trx_list, try_list):
        self.dataset = dataset
        self.idxs = idxs
        self.trx_list, self.try_list = trx_list, try_list
        self.X = torch.cat([self.dataset[0][self.idxs], self.trx_list])
        self.Y = torch.cat([self.dataset[1][self.idxs], self.try_list])
        # print("Y",self.Y[0:len(self.dataset[0][self.idxs])])
        #print(self.Y)
        #print(self.Y.shape)

    def __len__(self):
        return len(self.idxs)+len(self.trx_list)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y[idx]

        return img, label

class Client():
    def __init__(self, data_client, trainset, trx_list, try_list, args, device):
        self.data_client = data_client
        self.args = args
        self.device = device
        self.trainset = trainset
        self.loss = None
        self.trx_list, self.try_list =  trx_list, try_list
        self.cd = CustomDataset(self.trainset, self.data_client, self.trx_list, self.try_list)
        self.bcd = utils.CustomDataset(self.trx_list, self.try_list)
        if args.B == 8:
            self.bs = len(trainset)
        else:
            self.bs = args.B
        self.data_loader = torch.utils.data.DataLoader(self.cd, batch_size=self.bs,
                                                shuffle=True)
        self.bcd_loader = torch.utils.data.DataLoader(self.bcd, batch_size=self.bs,
                                                shuffle=True)
        if args.gpu == "gpu":
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        if args.model == 'nn':
            self.model_local = models.MP(28*28,200,10)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
         
        if args.model == 'cnn':
            self.model_local = models.CNN_MNIST()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.1, momentum=0.5)
        
        if args.model == 'resnet':
            #self.model_local = resnet18(num_classes=10)
            self.model_local = models.ResNet(18)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.01, momentum=0.5)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def client_update(self):
        print(len(self.cd))
        print(len(self.bcd))
        self.model_local.to(self.device)
        self.model_local.train()
        self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.1, momentum=0.5)
        for epoch in range(self.args.E):
            self.model_local.train()
            running_loss = 0.0
            running_acc = 0.0
            for index,data in enumerate(self.data_loader):
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                # for param in client.model_local.parameters():
                #    param.grad = None
                if self.args.model == 'nn':
                    inputs = inputs.flatten(1)
                outputs = self.model_local(inputs)
                pred = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
                acc = accuracy(pred,labels)
                running_acc+=acc
            print("epoch:",epoch)
            print("train-acc:", running_acc/(index+1))
            print("train-loss:", running_loss/(index+1))

            with torch.no_grad():
                self.model_local.eval()
                running_loss = 0.0
                running_acc = 0.0
                for index,data in enumerate(self.bcd_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model_local(inputs)
                    pred = torch.argmax(outputs, dim=1)
                    # print("pred-local",pred)
                    #print("img",inputs[0])
                    loss = self.criterion(outputs, labels)

                    running_loss+=loss.item()
                    acc = accuracy(pred,labels)
                    running_acc+=acc

            print("epoch:",epoch)
            print("bck-acc:", running_acc/(index+1))
            print("bck-loss:", running_loss/(index+1))
            


            
        # return self.model_local, loss, tr_acc, tr_poison_acc
        return self.model_local
