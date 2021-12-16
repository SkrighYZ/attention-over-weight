#!/bin/bash


# python train_new_task_aow.py --dataset cifar100 --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
# python train_new_task_aow.py --dataset cifar100 --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2
# python train_new_task_aow.py --dataset cifar100 --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --res 
python train_new_task_aow.py --dataset cifar100 --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 1

python train_new_task_aow.py --dataset svhn --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset svhn --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2

python train_new_task_aow.py --dataset gtsrb --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset gtsrb --wd 1e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2


#########################

python train_new_task_aow.py --dataset aircraft --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset aircraft --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2

python train_new_task_aow.py --dataset dtd --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset dtd --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2

python train_new_task_aow.py --dataset vgg-flowers --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset vgg-flowers --wd 2e-3 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2


#########################


python train_new_task_aow.py --dataset omniglot --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset omniglot --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2

python train_new_task_aow.py --dataset ucf101 --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset ucf101 --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2

python train_new_task_aow.py --dataset daimlerpedcls --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2 --mode original
python train_new_task_aow.py --dataset daimlerpedcls --wd 5e-4 --source ../models/pretrained/imnet.t7 --expdir ../results --batch_size 128 --att_factor 2




