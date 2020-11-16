import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class model_disc(nn.Module):

    def __init__(self, output_class):

        super(model_disc, self).__init__()
        
        pad = math.ceil(max((math.ceil(32 / 2) - 1) * 2 + (3 - 1) * 1 + 1 - 32, 0) / 2)

        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, padding=pad, bias=True)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, padding=pad, bias=True)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x, 0.1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x, 0.1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x, 0.1)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x, 0.1)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x, 0.1)

        x = self.fc(x)
        
        return x
        
    def relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return torch.where(x < 0.0, leakiness * x, x)

    def global_avg_pool(self, x):
        assert len(x.size()) == 4
        return torch.mean(x, (2,3))
