from torch.utils.data import Dataset
import torch
import models
import torch.optim as optim
from torchvision.models import resnet18

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

        return img, label

class Client():
    def __init__(self, data_client, trainset, args, device):
        self.data_client = data_client
        self.args = args
        self.device = device
        self.trainset = trainset
        self.loss = None
        self.cd = CustomDataset(self.trainset, self.data_client)
        if args.B == 8:
            self.bs = len(trainset)
        else:
            self.bs = args.B
        self.data_loader = torch.utils.data.DataLoader(self.cd, batch_size=self.bs,
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
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.01, momentum=0.5)
        
        if args.model == 'resnet':
            self.model_local = resnet18(num_classes=100)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model_local.to(device)
            self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.01, momentum=0.5)
        
    def load_model(self, model_global):
        self.model_local.load_state_dict(model_global.state_dict())

    def client_update(self):
        self.model_local.to(self.device)
        self.model_local.train()
        self.optimizer = optim.SGD(self.model_local.parameters(), lr=0.01, momentum=0.5)
        for epoch in range(self.args.E):
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
        return self.model_local, loss