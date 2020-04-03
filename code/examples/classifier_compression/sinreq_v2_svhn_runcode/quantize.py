import numpy as np
import scipy
import pickle
#import analysis
#import load 

def quantize_1(data, n_bit):
	n_bit = n_bit + 1
	mul = np.multiply(data, 2**(n_bit-1))
	mul_round = np.round(mul)
	data_q = (mul_round.astype(np.float32)) / (2**(n_bit-1))
	#print(data_q)
	#data_q = data_q1 - 2**-(n_bit)
	return data_q


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

# mid-rise uniform quantization 
def quantize_44(data, n_bit):
	mul = np.multiply(data, 2**(n_bit-1))
	mul_round = np.floor(mul)
	data_q = (mul_round.astype(np.float32)) / (2**(n_bit-1))
	#print(data_q)
	data_q = data_q + 2**-(n_bit)
	return data_q

'''
n_bins = 100
path_params = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
path_noise = '/home/ahmed/projects/NN_quant/results/networks/alexnet/480.0_81.2092'
weights, biases = load.get_netparams(path_params)
noise = load.get_error(path_noise)
weights_sub_q = {}

for layer in weights:
	noise_sub = load.get_suberror(noise, 'w', [layer])
	y, x = np.histogram(noise_sub, bins=n_bins, normed=True)
	x = x[:-1]
	noise_sub_dist, noise_sub_params, noise_sub_dist_name = analysis.get_best_fit(noise_sub, x, y)
	# find boundary of distribution, i.e., the points that include for example 90% of weights
	cut_bit = 0
	area = 1
	while True:
		cut = 1. / 2**cut_bit
		area = noise_sub_dist.cdf(cut, noise_sub_params[0], noise_sub_params[1]) - noise_sub_dist.cdf(-cut, noise_sub_params[0], noise_sub_params[1])
		if area < 0.10:
			break
		cut_bit = cut_bit + 1
	cut_bit = cut_bit - 1
	print(layer + '\t' + (str)(cut_bit))
	weights_sub_q[layer] = quantize(weights[layer], cut_bit)

with open('/home/ahmed/projects/NN_quant/results/quantized_params/alexnet/quantized.pickle', 'w') as f:
	pickle.dump([weights_sub_q, biases], f)
'''

#It returns full network parameters (in numpy array format) and sorted layer names (only those layers containing weights)
def get_netparams(ckpt_path):
	data_dict = np.load(ckpt_path, encoding='latin1').item()
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	for op_name in data_dict:
		op_name2 = op_name.replace('-', '_')
		for param_name, data in data_dict[op_name].items():
			if param_name == 'weights':
				weights[op_name2] = data
			elif param_name == 'biases':
				biases[op_name2] = data
			elif param_name == 'mean':
				mean[op_name2] = data
			elif param_name == 'variance':
				variance[op_name2] = data
			elif param_name == 'scale':
				scale[op_name2] = data
			elif param_name == 'offset':
				offset[op_name2] = data
	netparams['weights'] = weights
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	return netparams

def quantize_network(path_params, layers, path_save, bits_q):
	print('#########################')
	print(bits_q)
	if '.ckpt' in path_params:
		#netparams = load.get_netparams(path_params)
		netparams = get_netparams(path_params)
	else:
		with open(path_params, 'rb') as f:
			netparams = pickle.load(f)

	if type(netparams) is dict:
		weights, biases = netparams['weights'], netparams['biases']
		mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	else:
		weights, biases = netparams[0:2]
		if len(netparams) > 2:
			mean, variance, scale, offset = netparams[2:6]
		else:
			mean, variance, scale, offset = {}, {}, {}, {}
	
	weights_sub_q = {}
	for i in range(0, len(layers)):
		#print(layers[i])
		if bits_q[i]==32:
			weights_sub_q[layers[i]] = weights[layers[i]]
		else:
			weights_sub_q[layers[i]] = quantize(weights[layers[i]], bits_q[i])
		#print(weights_sub_q[layers[i]].shape)
		#print(set(weights_sub_q[layers[i]].ravel()))
		#print(len(set(weights_sub_q[layers[i]].ravel())))
		#print('-------------------------------------------------')
		#weights_sub_q[layers[i]].astype(float,casting='safe')
	with open(path_save, 'wb') as f:
		pickle.dump([weights_sub_q, biases, mean, variance, scale, offset], f)

'''
# ALEXNET
# ----------
path_net = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py'
Wo_bar = '/home/ahmed/projects/NN_quant/results/weights_retrained/alexnet_conv2_retrained_test.pickle' # {Wo}' = retrained with new data (only 'conv2' trainable)
W1 = '/home/ahmed/projects/NN_quant/results/weights_retrained/alexnet_conv2_retrained_test_f.pickle' # = {W1} (retrained to minimize the q error)
#path_params = '/home/ahmed/projects/NN_quant/results/weights_retrained/alexnet_conv2.pickle'
path_params = W1
#path_save = '/home/ahmed/projects/NN_quant/results/weights_retrained/alexnet_conv2_7.pickle' # save {Wo}'q (quantized version) - then ll be used to check testing acc 
path_save = '/home/ahmed/projects/NN_quant/results/weights_retrained/alexnet_conv2_7_W1q.pickle' # save {W1}q (quantized version) - then ll be used to check testing acc 
layers = load.get_layers(path_net)
bits_q = [8] * len(layers)
bits_q[layers.index('conv2')] = 7
quantize_network(path_params, layers, path_save, bits_q)
print("quantization is done: {W} into {W}q ")
'''

'''
# RESTNET18
# ---------
layers = load.get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
#layer = 'res2a_branch2a'
path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
#path_params = '/home/ahmed/projects/NN_quant/results/normalized/resnet18/res2a_branch2a3_shift_04May.pickle' # normalized layer
path_params = path_ckpt
for i in range(0, len(layers)):
	path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/May12_resnet18_10_' + layers[i] + '_5_bits.pickle'
	#path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_10_' + layers[i] + '_8_shift.pickle'
	bits_q = [10] * len(layers)
	bits_q[layers.index(layers[i])] = 5
	print(layers[i])
	quantize_network(path_params, layers, path_save, bits_q)
'''
