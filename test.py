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
epoch_num = 200
center_num = 10
batch_size = 32
print_iter = 2000/(batch_size/4)
pick_up_training = 1
ckpt_name = "vanilla"
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
        correct = 0
        total = 0
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
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

else:
    net, optimizer, _, _ = load_ckpt(ckpt_dir, ckpt_name, net, optimizer)


class_correct = [0.0]*len(classes)
class_total = [0.0]*len(classes)
net.eval()
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i,ci in enumerate(c):
        label = labels[i]
        class_correct[label] += ci.item()
        class_total[label] += 1


for i, label in enumerate(classes):
    print('Accuracy of %5s : %2d %%' % (
        label, 100 * class_correct[i] / class_total[i]))


