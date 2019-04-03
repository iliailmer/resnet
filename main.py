# from .model.resnet34 import ResNet34
import torchvision.transforms as transforms
import torchvision
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm


class BasicBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 stride,
                 kernel_size=3,
                 padding=1,
                 ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False
                               )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False
                               )
        self.relu = nn.ReLU()
        if stride != 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, classes=10):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = self.create_res_block(64, 1, 3)
        self.block2 = self.create_res_block(128, 2, 4)
        self.block3 = self.create_res_block(256, 2, 6)
        self.block4 = self.create_res_block(512, 2, 3)
        self.linear = nn.Linear(512, classes)
        self.relu = nn.ReLU()

    def create_res_block(self, out_channels, stride, blocks):
        strides = [stride] + [1] * (blocks - 1)
        res_blocks = []
        for stride in strides:
            res_blocks.append(BasicBlock(self.in_channels,
                                         out_channels,
                                         stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*res_blocks)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


net = ResNet34()
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net.cuda()
if torch.cuda.is_available():
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    device = 'cuda'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=5e-4)
best_acc = 0


# Training (https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(
        'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (len(trainloader)), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (len(testloader)), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


import gc

if __name__ == '__main__':
    for epoch in range(0, 0 + 200):
        train(epoch)
        test(epoch)
        gc.collect()
        torch.cuda.empty_cache()
