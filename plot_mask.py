import pickle
import numpy as np
import os

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

plt.switch_backend('agg')


# plot 2d heatmap using seaborn.heatmap() method
def plot_heatmap(arr, mask_num, svdir):
    # data_set = np.random.rand( 10 , 10 )
    ax = sns.heatmap( arr , linewidth = 0.5 , cmap = 'Blues' )

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    plt.rcParams["figure.figsize"] = (7, 5)
    fontsize = 20
    
    plt.title( "Layer {}".format(mask_num+1), fontdict={'fontsize': fontsize})
    fname = 'mask_' + str(mask_num) + '.png' 
    plt.savefig(os.path.join(svdir, fname))
    #plt.show()


# load mask
mask_file = '../results/results/cifar100_channel_2_nores_80_100_120_mask.pkl'
mask = pickle.load(open(mask_file, 'rb'))

mask_numpy = [t.detach().numpy() for t in mask]

# average masks across batches
mask_dict = defaultdict()
num_batches = len(mask_numpy)/25

for k in range(25):
    total = np.zeros(mask_numpy[k].shape)
    for i in range(k, len(mask_numpy), 25):
        total += mask_numpy[i]
    mask_dict[k] = total/num_batches

# plot and save fig
svdir = '../results/plots'

if not os.path.isdir(svdir):
    os.mkdir(svdir)


num_layers = 25

for mask_num in range(num_layers):
    # resize mask
    l = len(mask_dict[mask_num])
    h = 16 # fixed height
    w = l//h

    resized_mask = mask_dict[mask_num].reshape(h,w)

    # plot
    fname = 'mask_' + str(mask_num) + '.png'
    plot_heatmap(resized_mask, mask_num, svdir)

