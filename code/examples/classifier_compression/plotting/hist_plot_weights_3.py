import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import seaborn as sns

mypath = '../weights_sin2Reg/cifar10/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
layers = []
for each in sorted(onlyfiles):
   layers.append(np.load(mypath+each).ravel())

#layers.append(np.load(mypath+'/svhn_q/svhn_features0_quantized_wrpn.npy').ravel())
#layers.append(np.load(mypath+'/svhn_q/svhn_classifier0_quantized_wrpn.npy').ravel())
#layers.append(mypath+'/cifar10_conv2_quantized_wrpn.npy')
print(onlyfiles)

# plot
f, axes = plt.subplots(1, 6, figsize=(25, 5), sharex=False)
color = "b"
#sns.set()
#sns.set(style="white", palette="bright", color_codes=True)
sns.set(palette="bright", color_codes=True)

#plt.ylabel('counts')
#sns.distplot( layers[0] , ax=axes[0], color=color, bins=100, kde=False, axlabel='epoch#')
left = -0.35
right = 0.35
plt.subplot(1, 6, 1)
_ = plt.hist(layers[0], 50)
#plt.xlim((left,right))
plt.subplot(1, 6, 2)
_ = plt.hist(layers[1], 50)
#plt.xlim((left,right))
plt.subplot(1, 6, 3)
_ = plt.hist(layers[2], 50)
#plt.xlim((left,right))
plt.subplot(1, 6, 4)
_ = plt.hist(layers[3], 50)
#plt.xlim((left,right))
plt.subplot(1, 6, 5)
_ = plt.hist(layers[4], 50)
#plt.xlim((left,right))
#plt.xlim((left,right))
#plt.xlim((-0.5, 0.5))
#plt.savefig('examples/classifier_compression/figs/fig_sin2_bits-44444_cf_'+str(cf)+'_lr_'+str(lr)+'_TMP.png')
plt.savefig('cifar10_sinreq-learn.png')
