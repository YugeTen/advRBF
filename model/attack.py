from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image


from utils import rm_dir, cuda, where

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_adv_val_min=-1, x_adv_val_max=1):
        x_adv = Variable(x.data, requires_grad=True) # assign x_adv with values of x
        h_adv = self.net(x_adv)  # h_adv: network response to perturbed input

        if targeted:
            cost = self.criterion(h_adv, y)
        else:
            cost = -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_adv_val_min, x_adv_val_max)

        h = self.net(x) # h: network response to original input
        h_adv = self.net(x_adv)  # h_adv: network response to perturbed input

        return x_adv, h_adv, h