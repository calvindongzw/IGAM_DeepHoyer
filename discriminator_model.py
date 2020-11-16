import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class model_cifar(nn.Module):

    def __init__(self, output_class):

        super(model_cifar, self).__init__()
        
        pad1 = math.ceil(max((math.ceil(32 / 1) - 1) * 1 + (3 - 1) * 1 + 1 - 32, 0) / 2)
        pad2 = math.ceil(max((math.ceil(32 / 2) - 1) * 2 + (3 - 1) * 1 + 1 - 32, 0) / 2)

        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=pad1)
        self.conv2 = nn.Conv2d(16, 160, 3, 1, padding=pad1)
        self.conv3 = nn.Conv2d(160, 160, 3, 1, padding=pad1)
        self.conv4 = nn.Conv2d(160, 320, 3, 2, padding=pad2)
        self.conv5 = nn.Conv2d(320, 320, 3, 1, padding=pad1)
        self.conv6 = nn.Conv2d(320, 640, 3, 2, padding=pad2)
        self.conv7 = nn.Conv2d(640, 640, 3, 1, padding=pad1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(160)
        self.bn3 = nn.BatchNorm2d(320)
        self.bn4 = nn.BatchNorm2d(640)

        #Finetune
        self.fc = nn.Linear(640 * 32 * 32, output_class)
        
    def forward(self, x):
        #Initialize
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x, 0.1)
        orig_x = x
        
        #Unit 1_0
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x, 0.1)
        x = self.conv3(x)

        orig_x = F.avg_pool2d(orig_x, 1, 1)
        pad = (0,0,0,0,(160 - 16) // 2, (160 - 16) // 2,0,0)
        orig_x = F.pad(orig_x, pad)
        x += orig_x


        #Unit 1_i
        for i in range(5):
            orig_x = x
            x = self.bn2(x)
            x = self.relu(x, 0.1)

            x = self.conv3(x)
            x = self.bn2(x)
            x = self.relu(x, 0.1)
            x = self.conv3(x)
            x += orig_x

        #Unit 2_0
        orig_x = x
        x = self.bn2(x)
        x = self.relu(x, 0.1)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x, 0.1)
        x = self.conv5(x)

        orig_x = F.avg_pool2d(orig_x, 2, 2)
        pad = (0,0,0,0,(320 - 160) // 2, (320 - 160) // 2,0,0)
        orig_x = F.pad(orig_x, pad)
        x += orig_x
    
        #Unit 2_i
        for i in range(5):
            orig_x = x
            x = self.bn3(x)
            x = self.relu(x, 0.1)

            x = self.conv5(x)
            x = self.bn3(x)
            x = self.relu(x, 0.1)
            x = self.conv5(x)
            x += orig_x

        #Unit 3_0
        orig_x = x
        x = self.bn3(x)
        x = self.relu(x, 0.1)

        x = self.conv6(x)
        x = self.bn4(x)
        x = self.relu(x, 0.1)
        x = self.conv7(x)

        orig_x = F.avg_pool2d(orig_x, 2, 2)
        pad = (0,0,0,0,(640 - 320) // 2, (640 - 320) // 2,0,0)
        orig_x = F.pad(orig_x, pad)
        x += orig_x

        #Unit 3_i
        for i in range(5):
            orig_x = x
            x = self.bn4(x)
            x = self.relu(x, 0.1)

            x = self.conv7(x)
            x = self.bn4(x)
            x = self.relu(x, 0.1)
            x = self.conv7(x)
            x += orig_x

        #Unit last
        x = self.bn4(x)
        x = self.relu(x, 0.1)
        #x = self.global_avg_pool(x)

        #logit
        x = self.fc(x)
        
        return x
        

    def relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return torch.where(x < 0.0, leakiness * x, x)

    def global_avg_pool(self, x):
        assert len(x.size()) == 4
        return torch.mean(x, (2,3))
