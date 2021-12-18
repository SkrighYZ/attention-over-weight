import pickle
import attention_model_utils
import json


datasets = ['aircraft',
'cifar100',
'daimlerpedcls',
'dtd',
'gtsrb',
'vgg-flowers',
'omniglot',
'svhn',
'ucf101']

accs = {'imagenet': 60.32}

results_dir = save_dir + '/checkpoints/'
model_appendix = '_channel_2_nores_80_100_120'
save_dir = '../results/'

for dataset in datasets:
	acc_file = pickle.load(open(results_dir+dataset+model_appendix+'_acc.pkl', 'rb'))
	acc = acc_file['acc']
	accs[dataset] = acc

vdd_score = attention_model_utils.vdd_score(accs)
print(vdd_score)


accs['vdd_score'] = vdd_score
json.dump((open(save_dir+'results'+model_appendix+'.json', 'w')))
