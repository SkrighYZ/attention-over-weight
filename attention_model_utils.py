import math
from collections import defaultdict

baseline_accuracies = {
'ImNet':59.87,
'Airc':60.34,
'C100':82.12,
'DPed':92.82,
'DTD':55.53,
'GTSR':97.53,
'Flwr':81.41,
'OGlt':87.69,
'SVHN':96.55,
'UCF':51.20
}

lwf_accuracies = {
'ImNet':59.87,
'Airc':61.15,
'C100':82.23,
'DPed':92.34,
'DTD':58.83,
'GTSR':97.57,
'Flwr':83.05,
'OGlt':88.08,
'SVHN':96.10,
'UCF':50.04
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
