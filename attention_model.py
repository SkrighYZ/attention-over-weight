# models.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
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


class MaskedConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 in_size=0, nb_tasks=10):
        super(MaskedConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_size = in_size


        if groups != 1:
            raise ValueError('Only supports group size 1 for now.')

        if bias:
            raise ValueError('Only supports 0 bias for now. The BN layer handles bias.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups     

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        x_channels = in_size*in_size*in_channels
        w_channels = torch.numel(self.weight) + out_channels if bias else torch.numel(self.weight)
        self.attn_dim = num_weights // 8
        self.attns = nn.ModuleList([AttnOverWeight(x_channels, w_channels, self.attn_dim) for i in range(nb_tasks)])

    def forward(self, input):

        task = config_task.task

        x = input.flatten(start_dim=1)

        # Attention should be performed separately on weights and bias
        # But we don't have bias anyways for now, so the first if will never be called
        if self.bias:
            wb = torch.cat([self.weight.flatten(), self.bias]).unsqueeze(0)
        else:
            wb = self.weight.flatten()
        masked_wb = self.attns[task](x, wb)

        batch_size = x.size(0)
        if self.bias:
            masked_w = masked_wb[:, :-self.out_channels].reshape(batch_size, self.out_channels, self.in_channels, *self.kernel_size)
            masked_b = masked_wb[:, -self.out_channels:]
        else:
            masked_w = masked_wb.reshape(batch_size, self.out_channels, self.in_channels, *self.kernel_size)
            masked_b = None

        # move batch dim into out_channels
        weights = masked_w.unsqueeze(0).view(-1, self.in_channels, *self.kernel_size) # (N*C_out, C_in, K_h, K_w)
        # move batch dim into in_channels
        x = x.view(1, -1, x.size(2), x.size(3)) # (1, N*C_in, H, W)
        # move batch dim into out_channels
        bias = masked_b.flatten() if self.bias else None  # (N*C_out, )

        out_grouped = F.conv2d(input, weights, bias, self.stride, self.padding, self.dilation, groups=batch_size)

        return out_grouped.view(batch_size, self.out_channels, out_grouped.size(2), out_grouped.size(3))

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


class AttnOverWeight(nn.Module):
    def __init__(self, x_channels, wb_channels, attn_dim):
        super(AttnOverWeight, self).__init__()

        self.attn_dim = attn_dim
        self.fc_q = nn.Linear(x_channels, attn_dim)
        self.fc_k = nn.Linear(wb_channels, attn_dim)
        self.fc_v = nn.Linear(wb_channels, attn_dim)
        self.fc_o = nn.Linear(attn_dim, wb_channels)
        self.gamma = nn.Parameter(torch.randn(1))

    # Suppose x, wb is flattened
    def forward(self, x, wb):
        q = self.fc_q(x)               # (N, attn_dim)
        k = self.fc_k(wb).unsqueeze(0)  # (1, attn_dim)
        v = self.fc_v(wb).unsqueeze(0)  # (1, attn_dim)

        attn_score = F.softmax(torch.bmm(q.unsqueeze(2) , k.unsqueeze(1)) / torch.sqrt(self.attn_dim))  # (N, attn_dim, attn_dim)
        attn = torch.bmm(attn_score, v.unsqueeze(2)).squeeze(2)     # (N, attn_dim)
        weighted_wb = self.fc_o(attn)     # (N, wb_channels)

        #mask_normalized = (mask - torch.min(mask, dim=1, keepdim=True)) / torch.max(mask, dim=1, keepdim=True)

        batch_size = x.size(0)
        expanded_wb = wb.unsqueeze(0).repeat(batch_size, 1)

        #masked_wb = self.gamma * expanded_wb + mask_normalized * expanded_wb  
        masked_wb = self.gamma * expanded_wb + weighted_wb		# (N, wb_channels)

        return masked_wb

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
        
    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        residual = self.avgpool(x)
        residual = torch.cat((residual, residual*0), 1)
        out = F.relu(y+residual)
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
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

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

