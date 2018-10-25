import torch
import torch.nn as nn
from src.rbf import RBF
import torch.nn.functional as F




class VanillaRBF(nn.Module):
    def __init__(self, center_num):
        super(VanillaRBF, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        mu = torch.cuda.FloatTensor(84, center_num).normal_().requires_grad_().unsqueeze(0)
        sigma = torch.cuda.FloatTensor(center_num).normal_().requires_grad_().unsqueeze(0)
        self.w = torch.cuda.FloatTensor(center_num).normal_().requires_grad_().unsqueeze(0)
        self.mu = mu.expand(-1, 84, center_num)
        self.sigma = sigma.expand(-1, center_num)
        self.center_num = center_num

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(-1)
        x_expanded = x.expand(-1, 84, 10)
        x_mu_diff = torch.add(x_expanded, -self.mu)
        x_mu_euc = torch.mul(x_mu_diff, x_mu_diff)
        sigma_squared = torch.mul(self.sigma, self.sigma)
        sigma_squared = sigma_squared.unsqueeze(1)
        sigma_squared = sigma_squared.expand(-1, 84, self.center_num)

        pre_exp = torch.div(-x_mu_euc, sigma_squared)
        exp = torch.exp(pre_exp)
        logits = self.w.view(1,1,self.center_num)*exp
        logits = torch.sum(logits, dim=-1)




        x = self.fc3(logits)
        return x



