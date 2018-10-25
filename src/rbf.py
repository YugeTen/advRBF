import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    def __init__(self, center_num):
        super(RBF, self).__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

        mu = nn.Parameter(torch.cuda.FloatTensor(84, center_num).normal_())
        sigma = nn.Parameter(torch.cuda.FloatTensor(center_num).normal_().requires_grad_().unsqueeze(0)
        self.w = torch.cuda.FloatTensor(center_num).normal_().requires_grad_().unsqueeze(0)
        self.mu = mu.expand(-1, 84, center_num)
        self.sigma = sigma.expand(-1, center_num)
        self.center_num = center_num

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        a = self.a.expand_as(x)
        b = self.b.expand_as(x)
        c = self.c.expand_as(x)
        return a * torch.exp((x - b)^2 / c)
