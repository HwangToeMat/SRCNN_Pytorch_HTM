import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):

    def __init__(self,i_ = 1, c1 = 64, c2=32, c3=1, k1 = 9, k2 = 1, k3 = 5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(i_, c1, k1)
        self.conv2 = nn.Conv2d(c1, c2, k2)
        self.conv3 = nn.Conv2d(c2, c3, k3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.conv3(x)
        return x
