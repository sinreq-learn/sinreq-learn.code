import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import seaborn as sns

mypath = '../weights_sin2Reg/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
layers = []
for each in sorted(onlyfiles):
   layers.append(np.load(mypath+each).ravel())

print(onlyfiles)

# plot
f, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=False)
color = "b"
sns.set()
sns.set(style="white", palette="bright", color_codes=True)

#plt.ylabel('counts')
#sns.distplot( layers[0] , ax=axes[0], color=color, bins=100, kde=False, axlabel='epoch#')
sns.distplot( layers[0] , ax=axes[0], color=color, bins=100, kde=False)
sns.distplot( layers[1] , ax=axes[1], color=color, bins=100, kde=False)
sns.distplot( layers[2] , ax=axes[2], color=color, bins=100, kde=False)
sns.distplot( layers[3] , ax=axes[3], color=color, bins=100, kde=False)
sns.distplot( layers[4] , ax=axes[4], color=color, bins=100, kde=False)
#plt.xlim((-0.5, 0.5))
#plt.savefig('examples/classifier_compression/figs/fig_sin2_bits-44444_cf_'+str(cf)+'_lr_'+str(lr)+'_TMP.png')
plt.savefig('w_svhn_f0.png')
