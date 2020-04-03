from __future__ import division
import numpy as np
import math
import scipy
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

#path_init = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnet_layers_quant_16-bits_15Oct.pickle'
#path_final = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnet_layers_quant_retrained_15Oct.pickle'


path_save = '../nn_quant_and_run_code/results/quantized/alexnet/alexnet'
path_init = path_save + '_train_1_layers_quant_retrained_31Oct_RL.pickle'

path_save = '../nn_quant_and_run_code/results/quantized/alexnet/alexnet'
path_final = path_save + '_train_1_test_retrained_quantized.pickle'
#path_final = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnet_train_1_test_retrained_quantized.pickle'

path_save = '../nn_quant_and_run_code/results/quantized/alexnet/alexnet'
path_final = path_save + '_train_0_layers_quant_retrained_31Oct.pickle'
path_final = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_110118_11.pickle'
#path_final = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnet_train_1_test_retrained_quantized.pickle'

def load_weights(param_path):
	with open(param_path, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		params = u.load()
	if type(params) is dict:
		weights_, biases_ = params['weights'], params['biases']
		mean_, variance_, scale_, offset_ = params['mean'], params['variance'], params['scale'], params['offset']
		print(params['mean'])
	else:
		weights_, biases_ = params[0:2]
		if len(params) > 2:
			mean_, variance_, scale_, offset_ = params[2:6]
		else:
			mean_, variance_, scale_, offset_ = {}, {}, {}, {}
	return weights_

weights_init = load_weights(path_init)
weights_final = load_weights(path_final)

layer_name = 'fc6'
#layer_name = 'conv4'

layer = weights_init[layer_name].ravel()
import seaborn as sns 
sns.set()
plt.subplot(1, 2, 1)
_ = plt.hist(layer, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 10})


layer = weights_final[layer_name].ravel()
plt.subplot(1, 2, 2)
_ = plt.hist(layer, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 10})

plt.savefig('tmp_sin2_2_'+layer_name+'_0_5bits_103118.png')
