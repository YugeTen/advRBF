import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.vanilla import Vanilla
from model.vanilla_rbf import VanillaRBF
from utils import cuda, where
from pathlib import Path
from torch.autograd import Variable
from model.data_loader import get_loader
from torchvision.utils import save_image
from model.attack import Attack

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.center_num = args.center_num
        self.model_name = args.model_name
        self.load_ckpt = args.load_ckpt # boolean
        self.dataset = args.dataset
        self.D_out = args.D_out
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.print_iter = args.print_iter
        self.data_dir = Path(args.data_dir)
        self.random_seed = args.random_seed
        self.shuffle = args.shuffle
        self.test_size = args.test_size
        self.data_loader, self.classes = get_loader(self.data_dir,
                                                    self.batch_size,
                                                    self.dataset,
                                                    self.random_seed,
                                                    self.shuffle,
                                                    self.test_size)
        self.mode = args.mode
        self.global_epoch = 0
        self.global_iter = 0


        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.dataset)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.print_summary_ = False if self.mode == 'train' else True


        # Histories
        self.history = dict()
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        # Initialisation
        self.model_init()
        if self.load_ckpt:
            self.load_checkpoint()
        self.criterion = nn.CrossEntropyLoss()

        criterion = nn.CrossEntropyLoss()
        self.attack = Attack(self.net, criterion=criterion)

    def model_init(self):
        # GPU
        if self.model_name == 'vanilla':
            self.net = cuda(Vanilla(self.D_out), self.device)
        elif self.model_name == 'vanilla_rbf':
            self.net = cuda(VanillaRBF(self.D_out, self.center_num),
                            self.device)

        # init
        self.net.weight_init(_type='kaiming') # TODO: add to parse

        # optimizer
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],
                                betas=(0.5, 0.999))

    def save_checkpoint(self):
        states ={
            'iter':self.global_iter,
            'epoch':self.global_epoch,
            'history':self.history,
            'args':self.args,
            'model_states':self.net.state_dict(),
            'optim_states':self.optim.state_dict()
        }

        filepath = self.ckpt_dir.joinpath(self.model_name)
        torch.save(states, filepath.open('wb+'))
        print("===> saved checkpoint '{}' (iter {}, epoch {})\n".format(filepath, self.global_iter, self.global_epoch) )

    def load_checkpoint(self):
        filepath = self.ckpt_dir.joinpath(self.model_name)
        if filepath.is_file():
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states'])
            self.optim.load_state_dict(checkpoint['optim_states'])

            print("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(filepath))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        for epoch in range(self.epoch):
            self.global_epoch += 1
            correct = 0
            total = 0
            running_loss = 0.0
            print("#"*12+"\t Epoch %d \t"%(self.global_epoch)+"#"*12)

            for batch_idx, (inputs, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                inputs, labels = cuda(inputs,self.device), cuda(labels, self.device)
                self.optim.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()

                # stats:
                running_loss += loss.item() #TODO: cost += F.cross_entropy(logit, y, size_average=False).data[0]
                correct += torch.eq(outputs.max(1)[1], labels).float().sum().data.item()
                total += labels.size(0)

                # print:
                if batch_idx % self.print_iter == (self.print_iter -1):
                    print('[%d/%5d] loss: %.3f' %
                          (self.global_epoch, batch_idx + 1, running_loss / self.print_iter))
                    running_loss = 0.0

            train_accuracy = 100 * correct / total
            print('Training Accuracy: %d%%' % train_accuracy)
            self.test()
        print(" [*] Training Finished!")
        self.print_summary_=True
        self.test()

    def test(self):
        self.set_mode('eval')

        correct = 0
        total = 0
        cost = 0
        for batch_idx, (images, labels) in enumerate(self.data_loader['test']):
            images, labels = cuda(images, self.device), cuda(labels, self.device)
            outputs = self.net(images)

            correct += torch.eq(outputs.max(1)[1], labels).float().sum().data.item()
            cost += F.cross_entropy(outputs, labels, size_average=False).data.item()
            total += labels.size(0)

        test_accuracy = 100 * correct/total
        cost /= total

        # print results
        print('Testing Accuracy: %d%%\n' % test_accuracy)

        if self.history['acc'] < test_accuracy:
            self.history['acc'] = test_accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint()

        if self.print_summary_:
            print("Best test accuracy achieved at epoch %d: %d%%"
                  % (self.history['epoch'], self.history['acc']))
        else:
            self.set_mode('train')

    def attack(self, num_sample=100, target=-1, epsilon=0.03, alpha=2/255, iteration=None):
        self.set_mode('eval')

        x_true, y_true = self.sample_data(num_sample) # get 100 datapoints & groundtruths
        if isinstance(target, int) and (target in range(self.D_out)):
            y_target = torch.LongTensor(y_true.size()).fill_(target)
        else:
            y_target = None

        x_true = Variable(cuda(x_true, self.cuda), requires_grad=True)
        y_true = Variable(cuda(y_true, self.cuda), requires_grad=False)

        # set y_target as a variable (if there is one)
        if y_target:
            targeted = True
            y_target = Variable(cuda(y_target, self.cuda), requires_grad=False)
        else:
            targeted = False

        h = self.net(x_true)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, y_true).float().mean()
        cost = F.cross_entropy(h, y_true)

        # x_adv, h_adv, h = self.attack.i_fgsm(x_true, y_target, targeted, epsilon, alpha, iteration) \
        # if iteration else self.attack.fgsm(x_true, y_target, targeted, epsilon)
        x_adv, h_adv, h = self.attack.fgsm(x_true, y_target, targeted, epsilon)

        prediction_adv = h_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        print('[BEFORE] accuracy : {:.2f} cost : {:.3f}'.format(accuracy, cost))
        print('[AFTER] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv, cost_adv))

        self.set_mode('train')




    def sample_data(self, num_sample=100):
        """sample num_sample instances of data (duh)"""
        total = len(self.data_loader['test'].dataset) # TODO: does this work????
        seed = torch.FloatTensor(num_sample).uniform_(1, total).long()

        x = self.data_loader['test'].dataset.test_data[seed]
        x = self.scale(x.float().unsqueeze(1).div(255))
        y = self.data_loader['test'].dataset.test_labels[seed]

        return x, y


    def scale(self, image):
        return image.mul(2).add(-1)
























