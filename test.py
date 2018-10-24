import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from helper import load_ckpt

device = torch.device("cuda:0")
device_cpu = torch.device("cpu") #
# TODO: fix GPU

# parameters
epoch_num = 10
print_iter = 2000
pick_up_training = 1
ckpt_name = "vanilla"
data_dir = './data'
ckpt_dir = './ckpt'

# data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# CNN definition
class Vanilla(nn.Module):
    def __init__(self):
        super(Vanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Vanilla()

# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


best_test_accuracy = 0
pick_up_epoch = 0
net.to(device)

# train if no checkpoint
if not os.path.exists(os.path.join(ckpt_dir,ckpt_name)) or pick_up_training:
    if os.path.exists(os.path.join(ckpt_dir,ckpt_name)) and pick_up_training:
        net, optimizer, pick_up_epoch, best_test_accuracy = \
            load_ckpt(ckpt_dir, ckpt_name, net, optimizer)

    for epoch in range(pick_up_epoch,epoch_num):
        # training
        net.train()

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
            if i % print_iter == (print_iter-1):
                print('[%d/%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_iter))
                running_loss = 0.0

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



net, optimizer, _, _ = load_ckpt(ckpt_dir, ckpt_name, net, optimizer)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

