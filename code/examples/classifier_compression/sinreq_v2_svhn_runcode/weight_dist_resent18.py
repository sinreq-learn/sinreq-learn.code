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

date = '110618'
network_name = 'alexnet'
path_net = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.py'

HOME = '/home/ahmed/projects/rl_quantization/rl_quantization.code'
path_model = HOME + '/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.ckpt'

path_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_110618v2_0.pickle'
#path_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnet_train_1_test_retrained_quantized.pickle'
def load_netparams_ckpt(ckpt_path):
	data_dict = np.load(ckpt_path, encoding='latin1').item()
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	layer_names = []
	for op_name in data_dict:
		op_name2 = op_name.replace('-', '_')
		layer_names.append(op_name2)
		for param_name, data in data_dict[op_name].items():
			if param_name == 'weights':
				weights[op_name2] = data
			elif param_name == 'biases':	
				biases[op_name2] = data

	return weights

def load_weights(path_model):
	if '.ckpt' in path_model:
		params = np.load(path_model, encoding='latin1').item()
	else:
		with open(path_model, 'rb') as f:
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

def get_layers(path_net):
	layer_names = []
	with open(path_net, 'r') as infile:
		for line in infile:
			if ('.conv' in line) or ('.fc' in line):
				layer_names.append(line[line.index("name=") + 6 : line.rindex("'")])
	return layer_names


#weights_init = load_netparams_ckpt(path_model)
weights_init = load_weights(path_model)
layers = get_layers(path_net)

import seaborn as sns 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 0.1}	

num = 4
for i in range(1,num):
	layer_name = layers[i]
	layer = weights_init[layer_name].ravel()
	std = np.var(layer)
	print(layer_name)
	print(std)
	sns.set()
	plt.subplot(num-1, 2, i)
	_ = plt.hist(layer, 100)
	plt.xlim(-0.1, 0.1)
	#_ = plt.xlabel(layer_name)
	_ = plt.ylabel('Frequency')
	plt.tight_layout() # Or equivalently,  "plt.tight_layout()"
	#matplotlib.rc('font', **font)
	#plt.legend(layer_name)
	#matplotlib.rcParams.update({'font.size': 1})

for i in range(num,7):
	layer_name = layers[i]
	layer = weights_init[layer_name].ravel()
	std = np.var(layer)
	print(layer_name)
	print(std)
	print(num-1, 2, i)
	plt.subplot(num-1, 2, i)

	_ = plt.hist(layer, 100)
	plt.xlim(-0.1, 0.1)
	#_ = plt.xlabel(layer_name)
	_ = plt.ylabel('Frequency')
	plt.tight_layout() # Or equivalently,  "plt.tight_layout()"
	#matplotlib.rc('font', **font)
	#plt.legend(layer_name)
	#matplotlib.rcParams.update({'font.size': 1})
#plt.savefig(network_name+'_'+date+'.png')
plt.savefig(network_name+'_'+date+'.png', dpi=800)
