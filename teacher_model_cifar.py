import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        

        self.shortcut = nn.Sequential()
        
        if stride != 1 and in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(F.avg_pool2d(x, 2, 2), (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
        elif stride != 1:
            self.shortcut = LambdaLayer(lambda x: F.avg_pool2d(x, 2, 2))
        elif in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
        
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        print('x shape:', x.shape)
        print('out shape:', out.shape)
        #print(stride, self.in_ch, self.out_ch)
        print('shortcut:', self.shortcut(x).shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 160, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 320, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 640, num_blocks[2], stride=2)
        self.linear = nn.Linear(640, num_classes)

        self.apply(_weights_init)

    def global_avg_pool(self, x):
        assert len(x.size()) == 4
        return torch.mean(x, (2,3))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = self.linear(out)
        return out

class model_cifar(nn.Module):
    def __init__(self, output_class):
        super(model_cifar, self).__init__()

        self.resnet = ResNet(BasicBlock, [6, 6, 6], output_class)

    def forward(self, x):
        out = self.resnet(x)

        return out
