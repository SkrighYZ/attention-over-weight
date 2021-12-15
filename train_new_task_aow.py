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
import attention_model
import models
from attention_model import MaskedConv2d 
import os
import time
import argparse
import numpy as np

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
parser.add_argument('--source', default='../models/pretrained/imnet.t7', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--eval_batch_size', default=100, type=int, help='eval batch size')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')

parser.add_argument('--mode', default='channel', type=str, help='Mode of attention [channel | individual]')
parser.add_argument("--res", action='store_true', help="Whether to use residual on each weight")
parser.add_argument('--att_factor', default=2, type=int, help='Attention dimension factor')
args = parser.parse_args()

config_task.factor = args.factor
config_task.att_factor = args.att_factor
config_task.mode = args.mode
config_task.res = args.res
args.use_cuda = True

if type(args.dataset) is str:
    args.dataset = [args.dataset]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 


args.ckpdir = args.expdir + '/checkpoints/'
args.svdir  = args.expdir + '/results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

#####################################

# Prepare data loaders
train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(args.batch_size, args.eval_batch_size, args.dataset,args.datadir,args.imdbdir,True)
args.num_classes = num_classes


##########################################################################

# Load checkpoint and initialize the networks with the weights of a pretrained network
print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.source)
net_old = checkpoint['net']
store_data = []
for name, m in net_old.named_modules():
    if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
        store_data.append(m.weight.data)

net = attention_model.resnet26(num_classes)
element = 0
for name, m in net.named_modules():
    if isinstance(m, MaskedConv2d) and (m.kernel_size[0]==3):
        print(name, m)
        m.weight.data = store_data[element]
        element += 1

store_data = []
store_data_bias = []
store_data_rm = []
store_data_rv = []
names = []

for name, m in net_old.named_modules():
    if isinstance(m, nn.BatchNorm2d) and 'bns.' in name:
        print(name)
        names.append(name)
        store_data.append(m.weight.data)
        store_data_bias.append(m.bias.data)
        store_data_rm.append(m.running_mean)
        store_data_rv.append(m.running_var)


for id_task in range(len(num_classes)):
    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'bns.'+str(id_task) in name:
            m.weight.data = store_data[element].clone()
            m.bias.data = store_data_bias[element].clone()
            m.running_var = store_data_rv[element].clone()
            m.running_mean = store_data_rm[element].clone()
            element += 1

del net_old


##########################################################################


start_epoch = 0
best_acc = 0  # best test accuracy
results = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
all_tasks = range(len(args.dataset))
np.random.seed(1993)

net.cuda()
cudnn.benchmark = True


# Freeze convolution layers
for name, m in net.named_modules():
    if isinstance(m, MaskedConv2d):
        m.weight.requires_grad = False


args.criterion = nn.CrossEntropyLoss()
optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


print("Start training")
for epoch in range(start_epoch, start_epoch+args.nb_epochs):
    training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args)
    st_time = time.time()
    
    # Training and validation
    train_acc, train_loss = utils_pytorch.train(epoch, train_loaders, training_tasks, net, args, optimizer)
    test_acc, test_loss, best_acc = utils_pytorch.test(epoch,val_loaders, all_tasks, net, best_acc, args, optimizer)
        
    # Record statistics
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results[0:2,epoch,current_task] = [train_loss[i],train_acc[i]]
    for i in all_tasks:
        results[2:4,epoch,i] = [test_loss[i],test_acc[i]]
    res = 'res' if config_task.res else 'nores'
    np.save(args.svdir+'-'.join(args.dataset)+'_'+'_'.join([str(config_task.att_factor), res, str(args.step1), str(args.step2), str(args.nb_epochs)]), results)
    print('Epoch lasted {0}'.format(time.time()-st_time))

