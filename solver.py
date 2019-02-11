import os, pickle
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.vanilla import Vanilla
from model.vanilla_rbf import VanillaRBF
from utils import cuda, where, class_name_look_up
from pathlib import Path
from torch.autograd import Variable
from model.data_loader import get_loader
from torchvision.utils import save_image
from model.attack import Attack
from collections import defaultdict


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = (args.cuda and torch.cuda.is_available())
        self.epochs = args.epochs
        self.center_num = args.center_num
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.D_out = args.D_out
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.print_iter = args.print_iter
        self.data_dir = Path(args.data_dir)
        self.random_seed = args.random_seed
        self.shuffle = args.shuffle
        self.test_size = args.test_size

        self.mode = args.mode
        self.global_epoch = 0

        self.ckpt_dir = args.ckpt_dir
        self.stat_dir = args.stat_dir

        # Initialisation
        self.name_save_dir() # self.save_name, self.normalise_vector, self.data_dir

        self.model_stat_dir = os.path.join(self.stat_dir, self.save_name)
        self.model_ckpt_dir = os.path.join(self.ckpt_dir, self.save_name)
        self.model_stat_path = os.path.join(self.stat_dir, self.save_name, '{}'.format(str(self.trial).zfill(2)))
        os.makedirs(self.model_ckpt_dir, exist_ok=True)
        os.makedirs(self.model_stat_dir, exist_ok=True)

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.print_summary_ = False if self.mode == 'train' else True

        self.load_from = args.load_from

        # FGSM parameters
        self.num_sample = args.num_sample
        self.target = args.target
        self.epsilon = args.epsilon
        self.alpha = args.alpha
        self.iteration = args.iteration

        # misc
        self.trial = args.trial

        # TODO: fix
        self.classes = class_name_look_up(self.data_dir, self.dataset)
        self.data_loader = get_loader(self.data_dir,
                                      self.batch_size,
                                      self.dataset,
                                      self.random_seed,
                                      self.shuffle,
                                      self.trial)



        # Initialisation
        self.model_init()
        self.load_checkpoint()
        self.criterion = nn.CrossEntropyLoss()

        criterion = nn.CrossEntropyLoss()
        self.attack_model = Attack(self.net, criterion=criterion)

    def model_init(self):
        # GPU
        if self.model_name == 'vanilla':
            self.net = cuda(Vanilla(self.D_out), self.device)
        elif self.model_name == 'vanilla_rbf':
            self.net = cuda(VanillaRBF(self.D_out, self.center_num),
                            self.device)

        # init
        self.net.weight_init(_type='kaiming')

        # optimizer
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],
                                betas=(0.5, 0.999))
    def name_save_dir(self):

        self.save_name = '{}_{}_{}'.format(self.model_name, self.dataset, self.center_num, self.trial)


    def save_checkpoint(self, test_accuracy):
        states = {
            'trial': self.trial,
            'epoch': self.global_epoch,
            'acc': test_accuracy,
            'best_epoch': self.best_epoch,
            'best_acc': self.best_test_acc,
            'args': self.args,
            'model_states': self.net.state_dict(),
            'optim_states': self.optim.state_dict()
        }
        trial_str, epoch_str = str(self.trial).zfill(2), str(self.global_epoch).zfill(2)
        ckpt_filepath = Path(os.path.join(self.model_ckpt_dir, '{}_{}.ckpt'.format(trial_str, epoch_str)))
        pickle.dump(self.observation, open(self.model_stat_path, 'wb'))  # save observation
        torch.save(states, ckpt_filepath.open('wb+'))  # save ckpt
        best_str = 'BEST ' if self.is_best_ else '' # TODO: add best
        print("===> saved {}checkpoint '{}' (trial {}, epoch {})\n".format(best_str,
                                                                           ckpt_filepath,
                                                                           self.trial,
                                                                           self.global_epoch))

        if self.is_best_:  # save another copy 'best'
            best_ckpt_filepath = Path(os.path.join(self.model_ckpt_dir,
                                                   '{}best.ckpt'.format(self.trial)))
            torch.save(states, best_ckpt_filepath.open('wb+'))

    def load_checkpoint(self):
        # because make_dirs is called before this, model_ckpt_dir must exist
        ckpt_name = ''
        ckpt_dir_list = [dir for dir in os.listdir(self.model_ckpt_dir) if
                         '{}_'.format(str(self.trial).zfill(2)) in dir]

        # if ckpt_dir_list is empty or load_from == 'fresh'
        if (not ckpt_dir_list or self.load_from == 'fresh'):
            print("=> no checkpoint used/found for {} trial {}, training...".format(self.save_name, self.trial))
            self.observation = defaultdict(list)
            self.run_epochs = self.epochs
        else:
            # load ckpt
            if self.load_from == 'last':
                ckpt_name = sorted(ckpt_dir_list)[-1]
            elif self.load_from == 'best':
                ckpt_name = '{}best.ckpt'.format(self.trial)
            model_ckpt = Path(os.path.join(self.model_ckpt_dir, ckpt_name))
            print("=> loading checkpoint '{}'".format(model_ckpt))
            checkpoint = torch.load(model_ckpt.open('rb'))
            self.trial = checkpoint['trial']
            self.global_epoch = checkpoint['epoch']
            self.best_epoch = checkpoint['best_epoch']
            self.best_test_acc = checkpoint['best_acc']
            self.net.load_state_dict(checkpoint['model_states'])
            self.optim.load_state_dict(checkpoint['optim_states'])
            print("=> loaded %s checkpoint at %s. Testing accuracy: %d%%" %
                  (self.load_from, model_ckpt, checkpoint['acc']))

            # load stat
            f = pickle.load(open(self.model_stat_path, "rb"))
            self.observation = f
            if self.load_from == 'best':
                for key, item in self.observation.items():
                    self.observation[key] = self.observation[key][:self.best_epoch]

            self.run_epochs = self.epochs - self.global_epoch


    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        if self.mode == 'crossval':
            print("\n" + "#" * 17 + "\n \t Trial %s \t \n" % (self.trial) + "#" * 17 + "\n")

        for epoch in range(self.epochs):
            self.global_epoch += 1
            epoch_iter = 0
            correct = 0
            total = 0
            running_loss = 0.0
            print("#" * 12 + "\t Epoch %d [%s, %d]\t" %
                  (self.global_epoch, ", ".join(self.save_name.split('_')), self.trial) + "#" * 12)

            for batch_idx, (inputs, labels) in enumerate(self.data_loader['train']):
                epoch_iter += 1

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
            self.observation['train_loss'].append(running_loss / epoch_iter)
            self.observation['train_accuracy'].append(train_accuracy)

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

        self.observation['test_loss'].append(cost)
        self.observation['test_accuracy'].append(test_accuracy)
        # print results
        print('Testing Accuracy: %d%%\n' % test_accuracy)

        if self.best_test_acc < test_accuracy:
            self.is_best_ = True
            self.best_test_acc = test_accuracy
            self.best_epoch = self.global_epoch
        self.save_checkpoint(test_accuracy)

        self.set_mode('train')

    def attack(self):
        self.set_mode('eval')

        x_true, y_true = self.sample_data(self.num_sample) # get 100 datapoints & groundtruths
        if isinstance(self.target, int) and (self.target in range(self.D_out)):
            y_target = torch.LongTensor(y_true.size()).fill_(self.target)
        else:
            y_target = None

        x_true = Variable(cuda(x_true, self.device), requires_grad=True)
        y_true = Variable(cuda(y_true, self.device), requires_grad=False)

        # set y_target as a variable (if there is one)
        if y_target:
            targeted = True
            y = Variable(cuda(y_target, self.device), requires_grad=False)

        else:
            targeted = False
            y = y_true

        h = self.net(x_true)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, y_true).float().mean()
        cost = F.cross_entropy(h, y_true)

        x_adv, h_adv, h = self.attack_model.i_fgsm(x_true, y, targeted, self.epsilon, self.alpha, self.iteration) \
        if self.iteration else self.attack_model.fgsm(x_true, y, targeted, self.epsilon)
        # x_adv, h_adv, h = self.attack_model.fgsm(x_true, y_true, targeted, epsilon)

        prediction_adv = h_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        print('[BEFORE] accuracy : {:.0f}% cost : {:.3f}'.format(accuracy*100, cost))
        print('[AFTER] accuracy : {:.0f}% cost : {:.3f}'.format(accuracy_adv*100, cost_adv))

        self.set_mode('train')




    def sample_data(self, num_sample=100):
        """sample num_sample instances of data (duh)"""
        total = len(self.data_loader['test'].dataset) # get total number of test examples available
        seed = torch.FloatTensor(num_sample).uniform_(1, total).long() # randomly chooses 100 indices out of the "total" number of example

        x = self.data_loader['test'].dataset.test_data[seed]
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        x = self.scale(x.float().div(255))

        y_test = np.asarray(self.data_loader['test'].dataset.test_labels)
        y = torch.from_numpy(y_test[seed])

        return x, y


    def scale(self, image):
        return image.mul(2).add(-1)
























