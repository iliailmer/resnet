import torch
from torch import nn


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


def test():
    net = ResNet34()
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())

# test()
