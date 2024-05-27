import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class depthwise_conv(nn.Module):
    def __init__(self, ch_in, kernel_size):
        super(depthwise_conv, self).__init__()
        self.ch_in = ch_in
        self.kernel_size = kernel_size
        # self.ch_out = ch_out
        self.depth_conv = nn.Conv1d(ch_in, ch_in, kernel_size=kernel_size, padding=int((kernel_size-1)/2), groups=ch_in)



    def forward(self, x):
        x = self.depth_conv(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class MC(nn.Module):
    def __int__(self, ch_in):
        super(MC, self).__init__()
        self.conv1 = depthwise_conv(ch_in, 5)
        self.conv2 = depthwise_conv(ch_in, 3)
        self.conv3 = depthwise_conv(ch_in, 1)
        self.bn = nn.BatchNorm1d(ch_in)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.bn(x1)
        x2 = self.bn(x2)
        x3 = self.bn(x3)
        x = torch.cat([x1, x2, x3], 1)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_t = nn.AdaptiveAvgPool1d(1)  # 时间维度池化
        self.pool_f = nn.AdaptiveAvgPool1d(1)  # 特征维度池化
        self.sru = MC(inp)
        self.cru = GhostModule(inp, inp)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = h_sigmoid()

    def forward(self, x):
        identity = x

        # 假设 x 的维度是 [batch_size, channels, seq_length]
        n, c, s = x.size()

        # 对时间步和特征维度应用池化
        x_t = self.pool_t(x).squeeze(-1)  # 对时间维度池化
        x_f = self.pool_f(x.transpose(1, 2)).transpose(1, 2).squeeze(-1)  # 对特征维度池化

        # 合并池化后的特征
        y = torch.cat([x_t, x_f], dim=1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 调整维度以匹配输入维度，然后应用注意力机制
        y = y.unsqueeze(-1).expand_as(identity)
        out = identity * y

        return out