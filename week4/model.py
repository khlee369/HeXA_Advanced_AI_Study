import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(400, 100)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    Net = MNISTNet()
    x = torch.randn(128, 784)
    pred = Net(x)
    print(pred.shape)