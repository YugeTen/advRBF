import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    def __init__(self, D_in, center_num):
        super(RBF, self).__init__()


        self.mu = nn.Parameter(torch.cuda.FloatTensor(D_in, center_num).normal_())
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(center_num).normal_())
        self.w = nn.Parameter(torch.cuda.FloatTensor(center_num).normal_())
        self.center_num = center_num


    def forward(self, x):
        # x: batch x length
        # logits: batch x length x center
        # outputs: batch x length
        x = x.unsqueeze(-1)
        mu = self.mu.repeat(x.size(0), 1, 1)
        sigma = self.sigma.repeat(x.size(0), x.size(1), 1)
        w = self.w.view(1,1,self.w.size(0))
        logits = torch.exp(-(x-mu)**2/sigma**2)
        x = torch.sum(w*logits, dim=-1)

        return x

