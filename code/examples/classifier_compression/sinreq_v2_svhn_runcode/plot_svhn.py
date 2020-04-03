from __future__ import division
import numpy as np
import math
import scipy
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.pyplot import figure


#retrained_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_FIRST.pickle'
#retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/alexnetretrained_quantized_SECOND.pickle' 

#retrained_model = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained.pickle'
#retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/alexnet/retrained_quantized.pickle' 

#retrained_model = retrained_model_quantized = '/home/ahmed/projects/rl_quantization/rl_quantization.code/nn_quant_and_run_code/results/quantized/svhn_net/svhn_net_retrained.pickle' 
cost_factor = 0
#retrained_model = retrained_model_quantized = 'results_retrained_models/svhn_net/quantized/svhn_net_retrained_lambda_'+str(cost_factor)+'.pickle'
retrained_model = retrained_model_quantized = 'results_retrained_models/svhn_net/quantized/svhn_net_retrained_lambda_1_CONV.pickle'
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


layers_names = ['hidden1', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6', 'hidden7', 'hidden8', 'hidden9', 'hidden10', 'digit1', 'digit2', 'digit3', 'digit4', 'digit5', 'digit_length']

layer_name = 'hidden1'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
import seaborn as sns 

sns.set()
plt.figure(figsize=(20,5))
ax = plt.subplot(1, 9, 1)
#ax.patch.set_facecolor(bg_color)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
_ = plt.ylabel('Frequency')
matplotlib.rcParams.update({'font.size': 1})
#plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden2'
qbits = 4
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 2)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
plt.savefig('nips19_data/svhn_sin2_'+layer_name+'_fig.png')
# --------------------------------------------------------------
layer_name = 'hidden3'
qbits = 3
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 3)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden4'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 4)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden5'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 5)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden6'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 6)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden7'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 7)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden7'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 8)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
# --------------------------------------------------------------
layer_name = 'hidden9'
layer = weights_init[layer_name].ravel()
layer_q = quantize(layer, qbits)
ax = plt.subplot(1, 9, 9)
_ = plt.hist(layer, 100, color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#_ = plt.hist(layer_q, 100)
_ = plt.xlabel(layer_name)
#_ = plt.ylabel('Frequency')
#matplotlib.rcParams.update({'font.size': 7})
plt.xlim((-0.1, 0.1))
#plt.xticks(size = 20)
#plt.yticks(size = 20)
# --------------------------------------------------------------
#figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
#figure(figsize=(30,30))

matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
#matplotlib.axes.Axes.ticklabel_format(self, axis='both', style='sci', scilimits=None, useOffset=None, useLocale=None, useMathText=None)
plt.savefig('nips19_data/svhn_sin2_5feb20.png', dpi=1200)
#plt.savefig('nips19_data/svhn_sin2_29dec19.png', dpi=300)
