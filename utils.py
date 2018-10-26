import os, argparse
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import torch.nn as nn

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def preprocessing(data_dir, batch_size, dataset="cifar-10"):


    if dataset == "cifar-10":
        with open(os.path.join(data_dir, dataset + "-batches-py", "batches.meta"), "rb") as f:
            meta = pickle.load(f)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform)
        classes = meta['label_names']

    elif dataset == "cifar-100":
        with open(os.path.join(data_dir, dataset + "-python", "meta"), "rb") as f:
            meta = pickle.load(f)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                               download=True, transform=transform)
        classes = meta['fine_label_names']

    else:
        raise UnknownDatasetError()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    data_loader = dict()
    data_loader['train'] = trainloader
    data_loader['test'] = testloader
    return data_loader, classes


def load_ckpt(ckpt_dir, ckpt_name, net, optimizer):
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt_name))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    test_accuracy = checkpoint['accuracy']
    return net, optimizer, epoch, test_accuracy

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cuda(tensor,is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()