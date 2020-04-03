from __future__ import division
import numpy as np
import math
import scipy
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


#retrained_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_FIRST.pickle'
#retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnetretrained_quantized_SECOND.pickle' 

#retrained_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained.pickle'
#retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_quantized.pickle' 

retrained_model = retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/svhn_net/svhn_net_retrained.pickle' 


def load_weights(param_path):
	with open(param_path, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		params = u.load()
	if type(params) is dict:
		weights_, biases_ = params['weights'], params['biases']
		mean_, variance_, scale_, offset_ = params['mean'], params['variance'], params['scale'], params['offset']
		#print(params['mean'])
	else:
		weights_, biases_ = params[0:2]
		if len(params) > 2:
			mean_, variance_, scale_, offset_ = params[2:6]
		else:
			mean_, variance_, scale_, offset_ = {}, {}, {}, {}
	return weights_

weights_init = load_weights(retrained_model)
weights_final = load_weights(retrained_model_quantized)

#layer_name = 'conv4'
#layer_name = 'fc7'
layer_name = 'hidden4'

layer = weights_init[layer_name].ravel()
import seaborn as sns 
sns.set()
plt.subplot(1, 2, 1)
_ = plt.hist(layer, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 10})
plt.xlim((-0.1, 0.1))

layer = weights_final[layer_name].ravel()
plt.subplot(1, 2, 2)
_ = plt.hist(layer, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 10})
plt.xlim((-0.1, 0.1))

plt.savefig('tmp_sin2_'+layer_name+'_4bits_010919_run_22.png')
