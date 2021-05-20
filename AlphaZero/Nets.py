## This File Contains all the different Neural Network Architectures used and the Loss function

import torch
import torch.nn as nn
import torch.nn.functional as functions

## Dense Network
class Dense(nn.Module):
    def __init__(self):
        super(Dense,self).__init__()
        self.fc1 = nn.Linear(6*7,32)   #board-size hard-coded
        self.fc2 = nn.Linear(32,16)
        self.probhead = nn.Linear(16,7)
        self.valuehead = nn.Linear(16,1)
        self.soft = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = x.view(-1,6*7)
        x = functions.relu(self.fc1(x))
        x = functions.relu(self.fc2(x))

        #action probs
        P = self.soft(self.probhead(x))
        
        #value probs
        v = self.tanh(self.valuehead(x))

        return P,v


## Convolutional Network
class Conv(nn.Module):
    def __init__(self):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(336,150)
        self.fc2 = nn.Linear(150,60)
        self.probhead = nn.Linear(60,7)
        self.valuehead = nn.Linear(60,1)
        self.soft = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = x.view(-1,1,6,7)
        x = functions.relu(self.bn1(self.conv1(x)))
        x = x.view(-1,6*7*8)
        x = functions.relu(self.fc1(x))
        x = functions.relu(self.fc2(x))
        
        P = self.soft(self.probhead(x))

        v = self.tanh(self.valuehead(x))
        
        return P,v

## Loss Function
class Alphaloss(nn.Module):
    def __init__(self):
        super(Alphaloss,self).__init__()
    
    def forward(self,z,v,pi,P):     #Notation as per AlphaZero Paper
        value_error = (z - v) **2

        policy_error = -torch.matmul(pi,torch.log(P).T)   # gives the same result
        #policy_error = torch.sum(-pi*torch.log(P),1)
        
        return (value_error.view(-1)+policy_error).mean()