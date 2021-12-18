import math
from collections import defaultdict

baseline_accuracies = {
'imagenet':59.87,
'aircraft':60.34,
'cifar100':82.12,
'daimlerpedcls':92.82,
'dtd':55.53,
'gtsrb':97.53,
'vgg-flowers':81.41,
'omniglot':87.69,
'svhn':96.55,
'ucf101':51.20
}

lwf_accuracies = {
'imagenet':59.87,
'aircraft':61.15,
'cifar100':82.23,
'daimlerpedcls':92.34,
'dtd':58.83,
'gtsrb':97.57,
'vgg-flowers':83.05,
'omniglot':88.08,
'svhn':96.10,
'ucf101':50.04
}

ra_accuracies = {
'imagenet':59.23,
'aircraft':63.73,
'cifar100':81.31,
'daimlerpedcls':93.30,
'dtd':57.02,
'gtsrb':97.47,
'vgg-flowers':83.43,
'omniglot':89.82,
'svhn':96.17,
'ucf101':50.28
}

def vdd_score(test_domain_accuracies):
    vdd_score = 0
    gamma = 2
    for dataset in baseline_accuracies.keys():
        test_acc = test_domain_accuracies[dataset]
        base_err = (100-baseline_accuracies[dataset])/100
        max_err = base_err * 2
        test_err = (100-test_acc)/100
        if test_err > max_err:
            continue
        alpha = 1000*(max_err**(-gamma))
        domain_score = alpha*((max_err-test_err)**gamma)
        vdd_score += domain_score
    return vdd_score