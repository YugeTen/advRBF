import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from helper import load_ckpt, preprocessing
from src.vanilla import Vanilla
from src.vanilla_rbf import VanillaRBF

device = torch.device("cuda:0")

# parameters
epoch_num = 1
center_num = 10
batch_size = 32
print_iter = 2000/(batch_size/4)
pick_up_training = 0
ckpt_name = "vanilla_rbf"
data_dir = './data'
ckpt_dir = './ckpt'
dataset = "cifar-10"


trainloader, testloader, classes = preprocessing(data_dir, batch_size, dataset)


if ckpt_name == "vanilla":
    net = Vanilla()
elif ckpt_name == "vanilla_rbf":
    net = VanillaRBF(center_num)
net.to(device)
for p in net.state_dict():
    print(p)


# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(), lr=0.001)


best_test_accuracy = 0
pick_up_epoch = 0

# train if no checkpoint
if not os.path.exists(os.path.join(ckpt_dir,ckpt_name)) or pick_up_training:
    if os.path.exists(os.path.join(ckpt_dir,ckpt_name)) and pick_up_training:
        net, optimizer, pick_up_epoch, best_test_accuracy = \
            load_ckpt(ckpt_dir, ckpt_name, net, optimizer)

    for epoch in range(pick_up_epoch,epoch_num):
        # training
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0
        print("#"*12+"\t Epoch %d \t"%(epoch+1)+"#"*12)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            if i % print_iter == (print_iter-1):
                print('[%d/%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_iter))
                running_loss = 0.0
        train_accuracy = 100 * correct / total
        print('Training Accuracy: %d%%' % train_accuracy)


        # testing
        # net.to(device_cpu)
        correct = 0
        total = 0
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images, labels = images.to(device_cpu), labels.to(device_cpu)  #
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        print('Testing Accuracy: %d%%\n' % test_accuracy)

        # save best model
        if test_accuracy > best_test_accuracy:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
            }, os.path.join(ckpt_dir, ckpt_name))

    print('Finished Training')

    for p in net.state_dict():
        print(p)
else:
    net, optimizer, _, _ = load_ckpt(ckpt_dir, ckpt_name, net, optimizer)
    for p in net.state_dict():
        print(p)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
net.eval()
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    # print(outputs.size())
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    print(labels)
    c = (predicted == labels).squeeze()
    # print(c)
    for i,ci in enumerate(c):
        label = labels[i]
        class_correct[label] += ci.item()
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# # data preprocessing
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')