import torch
from torch import nn
from torch.nn import functional as F
from vision import ConvNet

class CNN_encoder(nn.Module):
    def __init__(self,output_dim):
        super(CNN_encoder, self).__init__()
        self.output_dim = output_dim
        self.enc = ConvNet(4)
        self.fc1 = nn.Linear(1024,300)
        self.fc2 = nn.Linear(300,50)
        self.fc3 = nn.Linear(50, self.output_dim)
    
    def forward(self,img):
        x = self.enc(img)
        #print(x.shape)
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_prob = F.softmax(self.fc3(x),dim=1)
        return y_prob