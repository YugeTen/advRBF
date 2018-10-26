import torch
import torch.nn as nn
from model.rbf import RBF
import torch.nn.functional as F
from utils import kaiming_init, xavier_init




class VanillaRBF(nn.Module):
    def __init__(self, D_out, center_num):
        super(VanillaRBF, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.rbf = RBF(84, center_num)

        self.fc3 = nn.Linear(84, D_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.rbf(x)

        x = self.fc3(x)
        return x

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())
        elif _type == 'xavier':
            for ms in self._modules:
                xavier_init(self._modules[ms].parameters())


