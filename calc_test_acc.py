import torch
import torch.nn as nn
import attention_model
from torch.autograd import Variable

import imdbfolder_coco as imdbfolder
import utils_pytorch
import torch.optim as optim


class Args:
    def __init__(self):
        self.ckpdir = '../results/checkpoints'


train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(128, 100, ['cifar100'], '../data/decathlon-1.0/', '../data/decathlon-1.0/annotations', True)

net = attention_model.resnet26(num_classes)
ckpt = torch.load('../results/cifar100_channel_2_nores_80_100_120_ckpt.pth')
net.load_state_dict(ckpt)

net.cpu()

args = Args()
args.lr = 0.1
args.wd = 1e-4
args.step1 = 80
args.step2 = 100
args.nb_epochs = 120
args.use_cuda = False
args.dataset = ['cifar100']
args.svdir = '../results/results'
args.criterion = nn.CrossEntropyLoss()
best_acc = 0
all_tasks = range(len(args.dataset))

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


test_acc, test_loss, best_acc = utils_pytorch.test(1, val_loaders, all_tasks, net, best_acc, args, optimizer)

print test_acc, test_loss, best_acc
