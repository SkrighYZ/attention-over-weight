# models.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import config_task
import math

def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Parameter)):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        m.weight.data.uniform_()
        m.bias.data.zero_()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MaskedConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 in_size=0, nb_tasks=10):
        super(MaskedConv2d, self).__init__()


        if groups != 1:
            raise ValueError('Only supports group size 1 for now.')

        if bias:
            raise ValueError('Only supports 0 bias for now. The BN layer handles bias.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.transposed = False
        self.output_padding = (0, 0)
        self.groups = groups     
        self.in_size = in_size

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size), requires_grad=False)
        self.register_parameter('bias', None)

        w_channels = torch.numel(self.weight)

        if config_task.mode == 'channel':
            self.attn_dim = self.in_channels
            self.attns = nn.ModuleList([AttnOverChannel(self.in_channels, self.out_channels, self.kernel_size[0], self.attn_dim) for i in range(nb_tasks)])
        elif config_task.mode == 'individual':
            self.attn_dim = self.in_channels  // 8  # may try different values later
            self.attns = nn.ModuleList([AttnOverWeight(self.in_channels, w_channels, self.attn_dim) for i in range(nb_tasks)])

    def forward(self, input):

        task = config_task.task

        batch_size = input.size(0)

        x = input.view(batch_size, self.in_channels, -1).permute(0, 2, 1)

        # Attention should be performed separately on weights and bias
        # We don't have bias for now
        w = self.weight
        masked_w = self.attns[task](x, w)

        if masked_w.ndim > 1:
            maked_w = masked_w.view(batch_size, self.out_channels, self.in_channels, *self.kernel_size)

            # move batch dim into out_channels
            weights = masked_w.unsqueeze(0).view(-1, self.in_channels, *self.kernel_size) # (N*C_out, C_in, K, K)
            # move batch dim into in_channels
            x = input.view(1, -1, input.size(2), input.size(3)) # (1, N*C_in, H, W)

            out_grouped = F.conv2d(x, weights, None, self.stride, self.padding, self.dilation, groups=batch_size)
            output = out_grouped.view(batch_size, self.out_channels, out_grouped.size(2), out_grouped.size(3))

        else:
            weights = masked_w.view(self.out_channels, self.in_channels, *self.kernel_size)
            output = F.conv2d(input, weights, None, self.stride, self.padding, self.dilation)
        
        return output


    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', in_size={in_size}'
        s += ', attn_dim={attn_dim}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class AttnOverChannel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, attn_dim):
        super(AttnOverChannel, self).__init__()

        self.attn_dim = attn_dim

        # May change to 1x1 Conv later
        self.fc_q = nn.Linear(in_channels, attn_dim)
        self.fc_k = nn.Linear(in_channels*kernel_size*kernel_size, attn_dim)
        self.fc_v = nn.Linear(in_channels*kernel_size*kernel_size, attn_dim)
        self.fc_o = nn.Linear(attn_dim, in_channels*kernel_size*kernel_size)
        self.gamma = nn.Parameter(torch.randn(1))

    # x shape - (N, HW, in_channels)
    # w shape - (out_channels, in_channels, kernel_size, kernel_size)
    def forward(self, x, w):
        batch_size = x.size(0)

        w = w.reshape(w.size(0), -1)     # (out_channels, in_channels*kernel_size*kernel_size)

        q = self.fc_q(x)     # (N, HW, attn_dim)
        k = self.fc_k(w.view(1, w.size(0), -1)).repeat(batch_size, 1, 1)    # (N, out_channels, attn_dim)
        v = self.fc_v(w.view(1, w.size(0), -1))   # (out_channels, attn_dim)

        # Take softmax along out_channels dim to get contribution distribution of each conv filter to each pixel in HW dim
        attn_score = torch.softmax(torch.bmm(q, k.transpose(1, 2)).mean(0)/math.sqrt(self.attn_dim), dim=1)  # (HW, out_channels)

        # Currently taking a mean along HW; may improve later
        attn_out = attn_score.mean(dim=0).unsqueeze(1) * v   # (out_channels, attn_dim)
        
        weighted_w = self.fc_o(attn_out).flatten()        # (w_channels, )

        masked_w = w.flatten() + torch.tanh(self.gamma) * weighted_w              # (w_channels, )

        return weighted_w


class AttnOverWeight(nn.Module):
    def __init__(self, x_channels, w_channels, attn_dim):
        super(AttnOverWeight, self).__init__()

        self.attn_dim = attn_dim

        # May change to 1x1 Conv later
        self.fc_q = nn.Linear(x_channels, attn_dim)
        self.fc_k = nn.Linear(1, attn_dim)
        self.fc_v = nn.Linear(1, attn_dim)
        self.fc_o = nn.Linear(attn_dim, 1)
        self.gamma = nn.Parameter(torch.randn(1))

    # x shape - (N, HW, x_channels)
    # w shape - (out_channels, in_channels, kernel_size, kernel_size)
    def forward(self, x, w):
        batch_size = x.size(0)

        w = w.flatten()     # (w_channels, )

        q = self.fc_q(x)     # (N, HW, attn_dim)
        k = self.fc_k(w.reshape(1, -1, 1)).repeat(batch_size, 1, 1)    # (N, w_channels, attn_dim)
        v = self.fc_v(w.reshape(1, -1, 1)).repeat(batch_size, 1, 1)   # (N, w_channels, attn_dim)

        # Take softmax along w_channels dim to get contribution distribution of each weight to each pixel in HW dim
        attn_score = torch.softmax(torch.bmm(q, k.transpose(1, 2))/math.sqrt(self.attn_dim), dim=2)  # (N, HW, w_channels)

        # Currently taking a mean along HW; may improve later
        attn_out = attn_score.mean(dim=1).unsqueeze(2) * v   # (N, w_channels, attn_dim)
        
        weighted_w = self.fc_o(attn_out).squeeze(2)         # (N, w_channels)
        expanded_w = w.reshape(1, -1).repeat(batch_size, 1) # (N, w_channels)

        masked_w = expanded_w + self.gamma * weighted_w        # (N, w_channels)

        return masked_w

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, nb_tasks=1):
        super(conv_task, self).__init__()
        self.conv = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
    
    def forward(self, x):
        task = config_task.task
        y = self.conv(x)
        y = self.bns[task](y)
        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nb_tasks=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_task(in_planes, planes, stride, nb_tasks)
        self.conv2 = conv_task(planes, planes, 1, nb_tasks)
        self.avgpool = nn.AvgPool2d(2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
        
    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        out = F.relu(y+self.shortcut(residual))
        return out


class ResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=[10]):
        super(ResNet, self).__init__()
        nb_tasks = len(num_classes)
        blocks = [block, block, block]
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.pre_layers_conv = conv_task(3, int(32*factor), 1, nb_tasks) 
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2, nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2, nb_tasks=nb_tasks)
        self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256*factor)),nn.ReLU(True)) for i in range(nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linears = nn.ModuleList([nn.Linear(int(256*factor), num_classes[i]) for i in range(nb_tasks)])         

        for m in self.modules():
            m.apply(weight_init)
    
    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride, nb_tasks=nb_tasks))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks))
        return nn.Sequential(*layers)

    def forward(self, x):
        task = config_task.task
        x = F.relu(self.pre_layers_conv(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end_bns[task](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linears[task](x)
        return x


def resnet26(num_classes=10, blocks=BasicBlock):
    return ResNet(blocks, [4,4,4], num_classes)


