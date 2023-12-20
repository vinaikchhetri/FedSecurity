import torch
import torch.nn as nn
import torch.nn.functional as F


class MP(torch.nn.Module): 
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MP, self).__init__()
        self.in_features = in_dim
        self.num_hiddens = hid_dim
        self.num_classes = out_dim
        
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.in_features, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_hiddens, bias=True),
            torch.nn.ReLU(True)
        )
        self.classifier = torch.nn.Linear(in_features=self.num_hiddens, out_features=self.num_classes, bias=True)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# class MP(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim):
#         super().__init__()
#         self.hl1 = nn.Linear(in_dim, hid_dim)
#         nn.init.xavier_uniform_(self.hl1.weight)
#         nn.init.zeros_(self.hl1.bias)
#         self.hl2 = nn.Linear(hid_dim, out_dim)
#         nn.init.xavier_uniform_(self.hl2.weight)
#         nn.init.zeros_(self.hl2.bias)
#         self.ReLU = nn.ReLU()

#     def forward(self, x):
#         x = self.ReLU(self.hl1(x))
#         x = self.ReLU(self.hl2(x))
#         return x

class CNN_MNIST(nn.Module):
    def __init__(self, in_channels=1, hidden_size=200, num_classes=10):
        super(CNN_MNIST, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=(self.hidden_channels * 2) * (7 * 7), out_features=512, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, 5)
    #     self.fc1 = nn.Linear(64 * 4 * 4, 512)
    #     self.fc2 = nn.Linear(512, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



