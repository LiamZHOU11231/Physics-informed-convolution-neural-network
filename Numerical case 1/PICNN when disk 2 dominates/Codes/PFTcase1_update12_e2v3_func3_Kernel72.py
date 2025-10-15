# ========================================================================================================
# 功能：建立网络模型-轻量化CNN
# 超参数总数：MATLAB里的神经网络分析器得出的结果是 56.9K。
# ========================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

pi = math.pi

class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (50, 5), 1)
        self.bn1   = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(50, 1), stride=1)

        self.conv2 = nn.Conv2d(16, 16, (50, 1), 1)
        self.bn2   = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(50, 1), stride=1)

        self.conv3 = nn.Conv2d(16, 16, (50, 1), 1)
        self.bn3   = nn.BatchNorm2d(num_features=16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(50, 1), stride=1)

        self.drop1 = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(16 * 730 * 1, 2)
        self.bn4 = nn.BatchNorm1d(num_features=2)

    def forward(self, x):
        batch_size_x = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.drop1(x)
        x = x.view(batch_size_x, -1)  # 展平
        x = self.linear1(x)
        x = self.bn4(x)

        return x


