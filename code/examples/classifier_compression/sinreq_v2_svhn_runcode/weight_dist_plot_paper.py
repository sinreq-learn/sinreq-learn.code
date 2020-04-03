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

#retrained_model = retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/svhn_net/svhn_net_retrained.pickle' 
cost_factor = 500
retrained_model = retrained_model_quantized = 'results_retrained_models/svhn_net/quantized/svhn_net_retrained_lambda_'+str(cost_factor)+'.pickle'
#retrained_model = retrained_model_quantized = 'models_svhn_sin2/svhn_net_retrained_sin2_run1.pickle'

def quantize(data, n_bit):
        n = float(2**(n_bit))
        #n = float(2**(n_bit) - 1)
        #n = float(2**(n_bit-1) - 1)
        data_q = np.round(data * n) / n
        q_error = np.average(np.abs(np.subtract(data, data_q)))
        print('number of bits: ',n_bit)
        #print(n_bit)
        print(q_error)
        return data_q

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

bg_color = 'grey'
qbits = 4
#layer_name = 'conv4'
#layer_name = 'fc7'
layer_name = 'hidden1'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
import seaborn as sns 
sns.set()
frame1 = plt.gca()
ax = plt.subplot(1, 3, 1)
#ax.patch.set_facecolor(bg_color)
_ = plt.hist(layer, 100)
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 1})
plt.xlim((-0.1, 0.1))
#plt.xticks([])
#plt.yticks([])
# --------------------------------------------------------------
layer_name = 'hidden4'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 3, 2)
#ax.patch.set_facecolor(bg_color)
_ = plt.hist(layer, 100)
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
#plt.xticks([])
#plt.yticks([])
#ax.spines['right'].set_color('black')
#ax.spines['top'].set_color('black')

# --------------------------------------------------------------
layer_name = 'hidden8'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 3, 3)
#ax.patch.set_facecolor(bg_color)
#ax.patch.set_edgecolor('red')

#ax.grid = True
#ax.edgecolor =  8

_ = plt.hist(layer, 100)
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
#plt.xticks([])
#plt.yticks([])
#ax.spines['bottom'].set_color('red')
#frame1.set_axis_bgcolor('white')
#frame1.patch.set_facecolor((1.0, 0.47, 0.42))

# --------------------------------------------------------------
plt.savefig('nips19_data/tmp_sin2_'+layer_name+'_fig1.png')
