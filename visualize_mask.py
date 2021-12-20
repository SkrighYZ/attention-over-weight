# train_new_task_adapters.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import attention_model_viz
import models
from attention_model import MaskedConv2d
import os
import time
import argparse
import numpy as np
import pickle

from torch.autograd import Variable

import imdbfolder_coco as imdbfolder
import config_task
import utils_pytorch
import sgd

parser = argparse.ArgumentParser(description='PyTorch Attention Weight Masks')
parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay for the classification layer')
parser.add_argument('--nb_epochs', default=120, type=int, help='nb epochs')
parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--expdir', default='../results', help='Save folder')
parser.add_argument('--datadir', default='../data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='../data/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--ckpt', default='cifar100_channel_2_nores_80_100_120_ckpt.pth', type=str, help='Trained model')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--eval_batch_size', default=100, type=int, help='eval batch size')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')

parser.add_argument('--mode', default='channel', type=str, help='Mode of attention [channel | individual | original]')
parser.add_argument("--res", action='store_true', help="Whether to use residual on each weight")
parser.add_argument('--att_factor', default=2, type=int, help='Attention dimension factor')
args = parser.parse_args()

config_task.factor = args.factor
config_task.att_factor = args.att_factor
config_task.mode = args.mode
config_task.res = args.res
args.use_cuda = False

if type(args.dataset) is str:
    args.dataset = [args.dataset]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir)


args.ckpdir = args.expdir + '/'
args.svdir  = args.expdir + '/results/'
args.pkl = args.svdir + args.ckpt.replace('ckpt.pth', 'mask.pkl')

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir)

#####################################

# Prepare data loaders
train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(args.batch_size, args.eval_batch_size, args.dataset,args.datadir,args.imdbdir,True)
args.num_classes = num_classes


##########################################################################
net = attention_model_viz.resnet26(num_classes)
checkpoint = torch.load(args.ckpdir + args.ckpt)
net.load_state_dict(checkpoint)

net.cpu()
cudnn.benchmark = True


# Pass training data through the model in eval mode
net.train()

training_tasks = range(len(args.dataset))

for itera in range(len(training_tasks)):
    i = training_tasks[itera]
    config_task.task = i
    for batch_idx, (inputs, _) in enumerate(train_loaders[i]):
        if args.use_cuda:
            inputs, _ = inputs.cuda(), _
        with torch.no_grad():
            inputs, _ = Variable(inputs), _ 
            _ = net(inputs)
        
        if batch_idx % 5 == 0:
            print '{} batches processed'.format(batch_idx)

pickle.dump(attention_model_viz.MASK, open(args.pkl, 'wb'))
