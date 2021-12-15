#!/bin/bash


python train_new_task_aow.py --dataset cifar100 --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128