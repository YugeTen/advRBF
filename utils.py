import os, argparse
import torch
import torchvision
import pickle
from pathlib import Path
import torch.nn as nn
import sys
import numpy as np
from PIL import Image



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

def rm_dir(dir_path):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        print('removed empty dir :',p)

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def calculate_normalisation_params(data_dir):
    img_dir = data_dir

    normalise_vector_dir = os.path.join(img_dir, 'normalise_vector')

    if os.path.exists(normalise_vector_dir):
        normalise_vector = pickle.load(open(normalise_vector_dir, "rb"))
    else:
        file_dir_list = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir)])  # all files under stimuli_dir
        print("Calculating normalisation parameters for data in {}...".format(img_dir))
        x = np.array([np.array(Image.open(f)) for f in file_dir_list if f.endswith('.png')])
        x = x / 255  # from [0, 255] to [0, 1]

        if len(x.shape) == 3:
            mean = np.mean(x)
            std = np.std(x)
            normalise_vector = [[mean], [std]]

        elif len(x.shape) == 4:
            normalise_vector = [[], []]
            for x_channel in np.rollaxis(x, 0):
                normalise_vector[0].append(np.mean(x_channel))
                normalise_vector[1].append(np.std(x_channel))
        else:
            raise Exception("Input images should have dimension 2 or 3, got {} instead".format(len(x.shape) - 1))

        pickle.dump(normalise_vector, open(normalise_vector_dir, 'wb'))
    return normalise_vector


def class_name_look_up(data_dir, dataset):
    if dataset == 'catvdog':
        class_names = ['cat', 'dog']
    elif dataset == 'cifar-10':
        with open(os.path.join(data_dir, dataset + "-batches-py", "batches.meta"), "rb") as f:
            meta = pickle.load(f)
        class_names = meta['label_names']
    elif dataset == 'cifar-100':
        with open(os.path.join(data_dir, dataset + "-batches-py", "batches.meta"), "rb") as f:
            meta = pickle.load(f)
        class_names = meta['fine_label_names']
    else:
        raise ('unknown dataset')
    return class_names