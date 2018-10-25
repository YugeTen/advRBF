import os
import torch
import torchvision
import torchvision.transforms as transforms
import pickle


def preprocessing(data_dir, batch_size, dataset="cifar-10"):

    if dataset == "cifar-10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform)

    elif dataset == "cifar-100":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    with open(os.path.join(data_dir,dataset+"-batches-py","batches.meta"), "rb") as f:
        meta = pickle.load(f)

    return trainloader, testloader, meta['label_names']


def load_ckpt(ckpt_dir, ckpt_name, net, optimizer):
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt_name))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    test_accuracy = checkpoint['accuracy']
    return net, optimizer, epoch, test_accuracy