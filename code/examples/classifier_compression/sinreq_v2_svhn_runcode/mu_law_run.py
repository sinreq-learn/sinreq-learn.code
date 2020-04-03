from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import math 
from math import exp, expm1
from mu_law_quantize import mu_law_quantize 
from mu_law_quantize import snr
import load 


path_net = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py'
path_params = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
path_save = '/home/behnam/results/weights_retrained/alexnet_W_muLAW_uniform.pickle'
#path_save = '/home/behnam/results/weights_retrained/alexnet_W_muLAW_norm.pickle'

# reading weights from files 
if '.ckpt' in path_params:
		params = load.get_netparams(path_params)
else:
	with open(path_params, 'r') as f:
		params = pickle.load(f)

# loading the weights into variables 
if type(params) is dict:
	weights_, biases_ = params['weights'], params['biases']
	if len(params) > 2:
		mean_, variance_, scale_, offset_ = params['mean'], params['variance'], params['scale'], params['offset']
	else:
		mean_, variance_, scale_, offset_ = {}, {}, {}, {}
else:
	weights_, biases_ = params[0:2]
	if len(params) > 2:
		mean_, variance_, scale_, offset_ = params[2:6]
	else:
		mean_, variance_, scale_, offset_ = {}, {}, {}, {}

# reading the layers and determining the #bits per layer 
layers = load.get_layers(path_net)
bits_q = [5] * len(layers)
bits_q[layers.index('conv2')] = 5

weights_q = {}
SNR = {}
MSE = {}
MU_layers = [11, 41, 81, 61, 41, 17, 25, 11]
for i in range(0, len(layers)):
	mu = MU_layers[i]
	mu = 0.5
	#x = weights_[layers[i]].ravel()
	x = weights_[layers[i]]
	Xmax = np.amax(x)
	Xmax = 1
	weights_q[layers[i]] = mu_law_quantize(x,bits_q[i],mu,Xmax)
	#x=np.array([1.2000, -0.2000, -0.500, 0.400, 0.890, 1.3000])
	#Xmax = 1.5
	#yq, xq = mu_law_quantize(x,2,mu,Xmax)
	SNR[layers[i]], MSE[layers[i]] = snr(weights_q[layers[i]],params['weights'][layers[i]])
	print(layers[i]) # prints layer name 
	print(weights_q[layers[i]].shape) # prints layer shape 
	#print(params['weights'][layers[i]].shape)
	print(snr(weights_q[layers[i]],params['weights'][layers[i]])) # prints SQNR and MSE 

with open(path_save, 'w') as f:
	pickle.dump([weights_q, biases_, mean_, variance_, scale_, offset_], f)

'''
# mu optimization per layer per bitwidth 
SNR = []
MSE = []
MU = []
layers = 'conv2'
MU_layers = [11, 41, 81, 61, 41, 17, 25, 11]
for mu in range(1,200,20):
		weights_q = weights_.copy() 
		x = weights_q[layers]
		Xmax = np.amax(x)
		Xmax = 1
		weights_q[layers] = mu_law_quantize(x,5,mu,Xmax)
		s, e = snr(weights_q[layers],params['weights'][layers])
		MU.append(mu)
		SNR.append(s)
		MSE.append(e)
		print(layers)
		print('mu= ', mu)
		print(snr(weights_q[layers],params['weights'][layers]))


#fig, ax1 = plt.subplots(figsize=(3, 3))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(MU, SNR, 'bo-')
ax2.plot(MU, MSE, 'ro-')

ax1.set_xlabel('mu')
ax1.set_ylabel('SNR', color='b')
ax2.set_ylabel('Quant. MSE', color='r')
ax2.set_yscale('log')
ind = SNR.index(max(SNR))
ax1.text(MU[ind], max(SNR), 'mu ='+ str(MU[ind]) +'',fontsize=16)

plt.title('Opt. Mu (@ 5 bits) for AlexNET/'+ layers +'')

plt.savefig('/home/behnam/results/images/'+ layers +'.png')

with open(path_save, 'w') as f:
	pickle.dump([weights_q, biases_, mean_, variance_, scale_, offset_], f)

print("mu-LAW quantization is DONE")

'''