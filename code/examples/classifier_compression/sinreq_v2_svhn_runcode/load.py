from __future__ import division
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp 
import os
import pickle
#import quantize 

#from .quantize import quantize_network
import sys
sys.path.append("..")
import quantize
from quantize import quantize_network
"""
import sys
sys.path.append("..")
import quantize
from quantize import quantize_network
"""

from tensorflow.python.ops import random_ops

def _initializer(shape, dtype=tf.float32, partition_info=None):
     return random_ops.random_normal(shape)

def init_netparams_tf(ckpt_path, trainable=False):
	data_dict = np.load(ckpt_path, encoding='latin1').item()
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	layer_names = []
	trainable_layers = []
	for op_name in data_dict:
		op_name2 = op_name.replace('-', '_')
		layer_names.append(op_name2)
		with tf.variable_scope(op_name2):
			for param_name, data in data_dict[op_name].items():
				if param_name == 'weights':
					if op_name2 in trainable_layers:
						weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
						weights[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
					biases[op_name2] = tf.get_variable(name='bias', shape=data.shape[-1], initializer=tf.zeros_initializer, trainable=trainable)
				elif param_name == 'biases':	
					if op_name2 in trainable_layers:
						biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
						biases[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
				elif param_name == 'mean':
					if op_name2 in trainable_layers:		
						mean[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#mean[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
						mean[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
				elif param_name == 'variance':
					if op_name2 in trainable_layers:		
						variance[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#variance[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
						variance[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
				elif param_name == 'scale':
					print(op_name2)
					if op_name2 in trainable_layers:		
						scale[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#scale[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
						scale[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
				elif param_name == 'offset':
					if op_name2 in trainable_layers:		
						offset[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						#offset[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
						offset[op_name2] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.random.normal, trainable=trainable)
	#print(len(layer_names))
	netparams['weights'] = weights
	print(len(weights))
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	return netparams

def load_netparams_tf(ckpt_path, trainable=False):
	data_dict = np.load(ckpt_path, encoding='latin1').item()
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	layer_names = []
	trainable_layers = []
	for op_name in data_dict:
		op_name2 = op_name.replace('-', '_')
		layer_names.append(op_name2)
		with tf.variable_scope(op_name2):
			for param_name, data in data_dict[op_name].items():
				if param_name == 'weights':
					if op_name2 in trainable_layers:
						weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
					biases[op_name2] = tf.get_variable(name='bias', shape=data.shape[-1], initializer=tf.zeros_initializer, trainable=trainable)
				elif param_name == 'biases':	
					if op_name2 in trainable_layers:
						biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'mean':
					if op_name2 in trainable_layers:		
						mean[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						mean[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
				elif param_name == 'variance':
					if op_name2 in trainable_layers:		
						variance[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						variance[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
				elif param_name == 'scale':
					print(op_name2)
					if op_name2 in trainable_layers:		
						scale[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						scale[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
				elif param_name == 'offset':
					if op_name2 in trainable_layers:		
						offset[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						offset[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
	#print(len(layer_names))
	netparams['weights'] = weights
	print(len(weights))
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	return netparams
'''
def load_netparams_tf(ckpt_path, trainable=False):
	data_dict = np.load(ckpt_path).item()
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	layer_names = []
	for op_name in data_dict:
		op_name2 = op_name.replace('-', '_')
		layer_names.append(op_name2)
		with tf.variable_scope(op_name2):
			for param_name, data in data_dict[op_name].iteritems():
				if param_name == 'weights':
					# controllling weights training per layer **
					if op_name2 == 'conv2' or op_name2 == 'conv4':
						weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
					else:
						weights[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'biases':
					if op_name2 == 'conv2' or op_name2 == 'conv4':
						biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=False)
					else:	
						biases[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'mean':
					mean[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'variance':
					variance[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'scale':
					scale[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
				elif param_name == 'offset':
					offset[op_name2] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=trainable)
	netparams['weights'] = weights
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	return netparams
'''

#To do: generalize for all networks (i.e., networks with paramters other than weights and biases)
def load_netparams_tf_q(param_path, trainable=False):
	with open(param_path, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		params = u.load()
		#params = pickle.load(f, encoding='latin1')

	#for key, value in params['scale'].iteritems():
	#		print key
	
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
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	trainable_layers = ['conv2','conv3'] 
	for layer in weights_:
		#with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('weights_q'):
			if layer in trainable_layers:
				weights[layer] = tf.get_variable(name=layer, initializer=tf.constant(weights_[layer]), trainable=True)			
			else:
				weights[layer] = tf.get_variable(name=layer, initializer=tf.constant(weights_[layer]), trainable=trainable)			
		with tf.variable_scope('biases_q'):
			if layer in biases_:
				if layer in trainable_layers:
					biases[layer] = tf.get_variable(name=layer, initializer=tf.constant(biases_[layer]), trainable=True)
				else:
					biases[layer] = tf.get_variable(name=layer, initializer=tf.constant(biases_[layer]), trainable=trainable)
			else:
				biases[layer] = tf.get_variable(name=layer, shape=weights_[layer].shape[-1], initializer=tf.zeros_initializer, trainable=trainable)
	for layer in mean_:
		with tf.variable_scope('mean'):
			mean[layer] = tf.get_variable(name=layer, initializer=tf.constant(mean_[layer]))
		with tf.variable_scope('variance'):
			variance[layer] = tf.get_variable(name=layer, initializer=tf.constant(variance_[layer]))
		with tf.variable_scope('scale'):
			scale[layer] = tf.get_variable(name=layer, initializer=tf.constant(scale_[layer]))
		with tf.variable_scope('offset'):
			offset[layer] = tf.get_variable(name=layer, initializer=tf.constant(offset_[layer]))
	netparams['weights'] = weights
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	for key, value in netparams['scale'].items():
			print(key)
	return netparams




def load_svhn_netparams_tf_q(path, trainable=False):
	""" reading the file (retrained model)"""
	with open(path, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		params = u.load()
		
	if type(params) is dict:
		print("reading model params dictionary ...")
		weights_, biases_ = params['weights'], params['biases']
		mean_, variance_, scale_, offset_ = params['mean'], params['variance'], params['scale'], params['offset']
	
	weights = {}
	biases = {}
	mean = {}
	variance = {}
	scale = {}
	offset = {}
	netparams = {}
	for layer in weights_:
		#with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('weights_q'):
			weights[layer] = tf.get_variable(name=layer, initializer=tf.constant(weights_[layer]), trainable=False)			
		with tf.variable_scope('biases_q'):
			biases[layer] = tf.get_variable(name=layer, initializer=tf.constant(biases_[layer]), trainable=False)
	for layer in mean_:
		with tf.variable_scope('mean'):
			mean[layer] = tf.get_variable(name=layer, initializer=tf.constant(mean_[layer]), trainable=False)
		with tf.variable_scope('variance'):
			variance[layer] = tf.get_variable(name=layer, initializer=tf.constant(variance_[layer]), trainable=False)
		with tf.variable_scope('scale'):
			scale[layer] = tf.get_variable(name=layer, initializer=tf.constant(scale_[layer]), trainable=False)
		with tf.variable_scope('offset'):
			offset[layer] = tf.get_variable(name=layer, initializer=tf.constant(offset_[layer]), trainable=False)
	netparams['weights'] = weights
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	return netparams

# here
def init_svhn_netparams_tf(ckpt_path, trainable=False):
	#data_dict = np.load(ckpt_path, encoding='latin1').item()
	data_dict = cp.list_variables(ckpt_path)
	#cp.load_variable(ckpt_path,'digit1/dense/bias')

	weights = {}  # kernel 
	biases = {} # bias
	mean = {}   # moving_mean
	variance = {} # moving_variance
	scale = {}  # beta (?)
	offset = {} # gamma (?)

	netparams = {}
	layer_names = []

	# get layers names 
	layers_names =[]
	layers_dict = {}
	for each in data_dict:
		words = each[0].split("/")
		if words[0] not in layers_names: 
			layers_names.append(words[0]) # save all unique layers names 
			layers_dict[words[0]] = [] # initialize a list for each layer

	for each in data_dict:
		words = each[0].split("/")
		layers_dict[words[0]].append(each[0])
	
	#print('layers_dict')
	#print(layers_dict)

	for layer_name in layers_dict:
		with tf.variable_scope(layer_name):
			for each in layers_dict[layer_name]:
				words = each.split("/")
				param_name = words[-1]

				data = cp.load_variable(ckpt_path, each)
				
				#print('#################: DEBUG')
				#print(each)
				#print(param_name)
				
				if param_name == 'kernel':
					weights[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#weights[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#weights[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)
				elif param_name == 'bias':	
					biases[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#biases[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#biases[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)
				elif param_name == 'moving_mean':
					mean[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#mean[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#mean[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)
				elif param_name == 'moving_variance':
					variance[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#variance[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#variance[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)
				elif param_name == 'beta':
					scale[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#scale[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#scale[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)
				elif param_name == 'gamma':
					offset[layer_name] = tf.get_variable(name=param_name, initializer=tf.random_normal(shape=data.shape), trainable=True)
					#offset[layer_name] = tf.get_variable(name=param_name, initializer=_initializer(data.shape), trainable=True)
					#offset[layer_name] = tf.get_variable(name=param_name, shape=data.shape[-1], initializer=tf.initializers.random_normal(shape=data.shape[-1]), trainable=True)

	#print(len(layer_names))
	netparams['weights'] = weights
	#print(len(weights))
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	#print(netparams['weights']['hidden1'])
	return netparams

def load_svhn_netparams_tf(ckpt_path, trainable=False):
	#data_dict = np.load(ckpt_path, encoding='latin1').item()
	data_dict = cp.list_variables(ckpt_path)
	#cp.load_variable(ckpt_path,'digit1/dense/bias')

	weights = {}  # kernel 
	biases = {} # bias
	mean = {}   # moving_mean
	variance = {} # moving_variance
	scale = {}  # beta (?)
	offset = {} # gamma (?)

	netparams = {}
	layer_names = []

	# get layers names 
	layers_names =[]
	layers_dict = {}
	for each in data_dict:
		words = each[0].split("/")
		if words[0] not in layers_names: 
			layers_names.append(words[0]) # save all unique layers names 
			layers_dict[words[0]] = [] # initialize a list for each layer

	for each in data_dict:
		words = each[0].split("/")
		layers_dict[words[0]].append(each[0])
	
	#print('layers_dict')
	#print(layers_dict)

	for layer_name in layers_dict:
		with tf.variable_scope(layer_name):
			for each in layers_dict[layer_name]:
				words = each.split("/")
				param_name = words[-1]

				data = cp.load_variable(ckpt_path, each)
				
				#print('#################: DEBUG')
				#print(each)
				#print(param_name)
				
				if param_name == 'kernel':
					weights[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
				elif param_name == 'bias':	
					biases[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
				elif param_name == 'moving_mean':
					mean[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
				elif param_name == 'moving_variance':
					variance[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
				elif param_name == 'beta':
					scale[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)
				elif param_name == 'gamma':
					offset[layer_name] = tf.get_variable(name=param_name, initializer=tf.constant(data), trainable=True)

	#print(len(layer_names))
	netparams['weights'] = weights
	#print(len(weights))
	netparams['biases'] = biases
	netparams['mean'] = mean
	netparams['variance'] = variance
	netparams['scale'] = scale
	netparams['offset'] = offset
	#print(netparams['weights']['hidden1'])
	return netparams

def get_error(path):
	if os.path.isdir(path):
		path = os.path.join(path, '')
		err_w = []
		for filename in os.listdir(path):
			if 'png' in filename:
				continue
			with open(path + filename, 'r') as f:
				err_w.append(pickle.load(f))
		return err_w
	else:
		with open(path, 'r') as f:
			return [pickle.load(f)]

		
def get_suberror(data, mode, sub):
	err_w = data
	if mode == 'all-w' or mode == 'all':
		err_w_all = []
		for i in range(len(err_w)):
			for key in err_w[i]:
				err_w_all = np.concatenate([err_w_all,  err_w[i][key].ravel()])
		if mode == 'all-w': return err_w_all
	if mode == 'all-b' or mode == 'all':
		err_b_all = []
		for i in range(len(err_b)):
			for key in err_b[i]:
				err_b_all = np.concatenate([err_b_all,  err_b[i][key].ravel()])
		if mode == 'all-b': return err_b_all
	if mode == 'all-lyr' or mode == 'all':
		err_lyr_all = []
		for i in range(len(err_lyr)):
			for key in err_lyr[i]:
				err_lyr_all = np.concatenate([err_lyr_all,  err_lyr[i][key].ravel()])
		if mode == 'all-lyr': return err_lyr_all
	if mode == 'all':
		return np.concatenate([np.concatenate([err_w_all, err_b_all]), err_lyr_all])
	
	if mode == 'w':
		data_main = err_w
	elif mode == 'b':
		data_main = err_b
	elif mode == 'lyr':
		data_main = err_lyr
	L = []
	R = []
	ret_value = []
	layer_name = sub[0]
	for i in range(1, len(sub)): L.append(sub[i][0]), R.append(sub[i][-1] + 1)
	dim = len(sub) - 1
	if dim == 0:
		for element in data_main: ret_value += list(element[layer_name].ravel())
	elif dim == 1:
		for element in data_main: ret_value += list(element[layer_name][L[0]:R[0]].ravel())
	elif dim == 2:
		for element in data_main: ret_value += list(element[layer_name][L[0]:R[0], L[1]:R[1]].ravel())
	elif dim == 3:
		for element in data_main: ret_value += list(element[layer_name][L[0]:R[0], L[1]:R[1], L[2]:R[2]].ravel())	
	elif dim == 4:
		for element in data_main: ret_value += list(element[layer_name][L[0]:R[0], L[1]:R[1], L[2]:R[2], L[3]:R[3]].ravel())	
	return ret_value

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

def get_subparams(data, sub):
	L = []
	R = []
	subparam = []
	layer_name = sub[0]
	for i in range(1, len(sub)): L.append(sub[i][0]), R.append(sub[i][-1] + 1)
	dim = len(sub) - 1
	if dim == 0:
		for element in data: subparam += list(element[layer_name].ravel())
	elif dim == 1:
		for element in data: subparam += list(element[layer_name][L[0]:R[0]].ravel())
	elif dim == 2:
		for element in data: subparam += list(element[layer_name][L[0]:R[0], L[1]:R[1]].ravel())
	elif dim == 3:
		for element in data: subparam += list(element[layer_name][L[0]:R[0], L[1]:R[1], L[2]:R[2]].ravel())	
	elif dim == 4:
		for element in data: subparam += list(element[layer_name][L[0]:R[0], L[1]:R[1], L[2]:R[2], L[3]:R[3]].ravel())	
	return subparam

def get_weights_dist(ckpt_addr, err_dir, save_addr):
	netparams = get_netparams(ckpt_addr)
	weights = netparams['weights'] 
	weights_noisy_array = []
	for dirpath, subdirs, fileList in os.walk(err_dir):
		for filename in fileList:
			addr = (os.path.join(dirpath, filename))
			with open(addr, 'r') as f:
				# we load all entire weights of the network (all layers) per single run/100runs
				err_w = pickle.load(f)
			weights_noisy_element = {}
			for layer in weights:
				weights_noisy_element[layer] = weights[layer] + err_w[layer]
			weights_noisy_array.append(weights_noisy_element)
	with open(save_addr, 'w') as f:
		pickle.dump(weights_noisy_array, f)

def get_layers(path_net):
	layer_names = []
	with open(path_net, 'r') as infile:
		for line in infile:
			if ('.conv' in line) or ('.fc' in line):
				layer_names.append(line[line.index("name=") + 6 : line.rindex("'")])
	return layer_names

def quantize_weights(weights1, n_bit):
	weights_ = weights1.copy()
	for layer in weights_:
		data = weights_[layer]
		mul = np.multiply(data, 2**(n_bit-1))
		mul_round = np.round(mul)
		data_q = (mul_round.astype(np.float32)) / (2**(n_bit-1))
		weights_[layer] = data_q
	return weights_

'''
ckpt_addr = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
err_dir = '/home/ahmed/projects/NN_quant/results/deltas/lenet'
save_addr = '/home/ahmed/projects/NN_quant/results/deltas'
'''

#'''
# =================================================================================
# Normalization check for RESNET18 
# ---------------------------------
def net_range_normalize(path_ckpt, path_save):
	#path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
	netparams = get_netparams(path_ckpt)
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	dict_shift_perc = {}
	for layer in weights:
		weights_temp = weights[layer].ravel()
		dict_shift_perc[layer] = []
		for n in range(1, 32):
			percent = ((-1/2**n < weights_temp) & (weights_temp < 1/2**n)).sum() / (len(weights_temp) + 0.)
			if percent < 0.7:
				break
			dict_shift_perc[layer].append((n, percent))

	#layers = get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
	#layers.remove('res2b_branch2a')
	#layers.remove('res5b_branch2b')
	#layers.remove('fc1000')
	shift = [5, 3]
	layers = ['res2a_branch1', 'res2a_branch2a']
	for layer in layers:
		for tuple in dict_shift_perc[layer]:
			print(tuple)
			if tuple[1] == 1.:
				continue
			netparams = get_netparams(path_ckpt)
			weights_normalized = netparams['weights']
			temp = []
			temp = weights_normalized[layer].ravel()
			temp[temp > 1/2**tuple[0]] = 1/2**tuple[0] - 1/2**32
			temp[temp < -1/2**tuple[0]] = -(1/2**tuple[0] - 1/2**32)
			shift = 2**(tuple[0]-1)
			#temp = temp * 2**(tuple[0]-1)
			#temp = temp*(2**(tuple[0]-1)) 
			#temp = [x * (2**(tuple[0]-1)) for x in temp]
			#print(type(weights_normalized[layer]))
			weights_normalized[layer]=np.multiply(weights_normalized[layer],shift)
			weights_normalized = quantize_weights(weights_normalized, 10)
			print(max(weights_normalized[layer].ravel()))
			#with open(path_save + layer + str(tuple[0]) + '.pickle', 'w') as f:
			with open(path_save + layer + str(tuple[0]) + '_shift_04May.pickle', 'w') as f:
				pickle.dump([weights_normalized, biases, mean, variance, scale, offset], f)
#'''

'''
# sequence: sensitivity analysis per layer, saturate, shift, under_quantize, inference(conv/fc), shift back. check ACC 
# RESNET18
def shift_layers(layers, shift, path_ckpt, path_save):
	netparams = get_netparams(path_ckpt)
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	weights_normalized = netparams['weights']
	#print(weights_normalized[layers[0]])
	for i in range(0, len(layers)):
		temp = []
		temp = weights_normalized[layers[i]].ravel()
		# saturate 
		weights_normalized[layers[i]][weights_normalized[layers[i]] > 1/2**shift[i]] = 1/2**shift[i] - 1/2**32
		weights_normalized[layers[i]][weights_normalized[layers[i]] < -1/2**shift[i]] = -(1/2**shift[i] - 1/2**32)
		# shift left 
		bin_shift = 2**(shift[i]-1)
		print(bin_shift)
		weights_normalized[layers[i]]=np.multiply(weights_normalized[layers[i]],bin_shift)
		print(max(weights_normalized[layers[0]].ravel()))
		print('------------')
	with open(path_save + '_layers_shift_05May.pickle', 'w') as f:
		pickle.dump([weights_normalized, biases, mean, variance, scale, offset], f)
	# under_quantize 
	path_params = path_save + '_layers_shift_05May.pickle' # normalized layer (shifted)
	path_save_q = path_save + '_layers_shift_quant_05May.pickle'
	#path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_layers_shift_and_quant.pickle'
	layers_sorted = get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
	bits_q = [16] * len(layers_sorted)
	for i in range(0, len(layers)):
		bits_q[layers_sorted.index(layers[i])] = 10-shift[i]+1
		print(bits_q)
	quantize_network(path_params, layers_sorted, path_save_q, bits_q)

layers = ['res2a_branch1', 'res2a_branch2a']
layers = ['res2a_branch1', 'res2a_branch2a']
shift = [2, 3]
path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
#path_save = '/home/ahmed/projects/NN_quant/results/normalized/resnet18/'
path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18'
path_save_q = path_save + '_layers_shift_quant_05May.pickle'

shift_layers(layers, shift, path_ckpt, path_save)
print('===========================')
print('DONE!')
print('===========================')

'''
def shift_layers(layers, shift, path_ckpt, path_save):
	netparams = get_netparams(path_ckpt)
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	weights_normalized = netparams['weights']
	#print(weights_normalized[layers[0]])
	for layer in layers:
		# saturate 
		#weights_normalized[layer][weights_normalized[layer] > 1/2**(shift[layer]+1)] = 1/2**(shift[layer]+1) - 1/2**32
		#weights_normalized[layer][weights_normalized[layer] < -1/2**(shift[layer]+1)] = -(1/2**(shift[layer]+1) - 1/2**32)
		weights_normalized[layer][weights_normalized[layer] > 1/2**(shift[layer])] = 1/2**(shift[layer]+1) - 1/2**32
		weights_normalized[layer][weights_normalized[layer] < -1/2**(shift[layer])] = -(1/2**(shift[layer]+1) - 1/2**32)
		# shift left 
		bin_shift = 2**(shift[layer])
		print(bin_shift)
		weights_normalized[layer]=np.multiply(weights_normalized[layer],bin_shift)
		print(layer)
		print(max(weights_normalized[layer].ravel()))
		print('------------')
	with open(path_save + '_layers_shift_10May.pickle', 'w') as f:
		pickle.dump([weights_normalized, biases, mean, variance, scale, offset], f)
	# under_quantize 
	path_params = path_save + '_layers_shift_10May.pickle' # normalized layer (shifted)
	#path_save_q = path_save + '_layers_shift_quant_res2a_branch2b-2sh-3bits_14May.pickle'
	
	# (2)/3
	path_save_q = path_save + 'resnet18_layers_shift_quant_res5a_branch1_1sh-4bits_14May.pickle'
	#path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_layers_shift_and_quant.pickle'
	layers_sorted = get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
	bits_q = [10] * len(layers_sorted)
	for layer in layers:
		#bits_q[layers_sorted.index('res2a_branch1')] = 9
		#bits_q[layers_sorted.index('res2a_branch2a')] = 8
		bits_q[layers_sorted.index(layer)] = 10-shift[layer]
	# (3)/3
	quantize_layers = ['res2a_branch1','res2a_branch2a','res2a_branch2b','res2b_branch2a', 'res2b_branch2b', 'res3a_branch1']
	for layer in quantize_layers:
		bits_q[layers_sorted.index(layer)] = 5-shift[layer]
	#bits_q[layers_sorted.index('res3a_branch2a')] = 8
	bits_q[layers_sorted.index('res3a_branch2b')] = 5
	bits_q[layers_sorted.index('res3b_branch2a')] = 4
	
	bits_q[layers_sorted.index('res3b_branch2b')] = 3
	bits_q[layers_sorted.index('res4a_branch1')] = 3

	bits_q[layers_sorted.index('res4a_branch2a')] = 6
	#bits_q[layers_sorted.index('res4a_branch2b')] = 8
	bits_q[layers_sorted.index('res4b_branch2a')] = 6
	bits_q[layers_sorted.index('res4b_branch2b')] = 4
	bits_q[layers_sorted.index('res5a_branch1')] = 4
	
	
	'''
	bits_q[layers_sorted.index('res3b_branch2b')] = 5
	bits_q[layers_sorted.index('res4a_branch1')] = 5
	bits_q[layers_sorted.index('res4a_branch2a')] = 5
	bits_q[layers_sorted.index('res4a_branch2b')] = 5
	bits_q[layers_sorted.index('res4b_branch2a')] = 5
	bits_q[layers_sorted.index('res4b_branch2b')] = 5
	bits_q[layers_sorted.index('res5a_branch1')] = 5
	bits_q[layers_sorted.index('res5a_branch2a')] = 5
	bits_q[layers_sorted.index('res5a_branch2b')] = 5
	bits_q[layers_sorted.index('res5b_branch2a')] = 5
	bits_q[layers_sorted.index('res5b_branch2b')] = 5
	'''
	print(bits_q)
	quantize_network(path_params, layers_sorted, path_save_q, bits_q)



'''
shift_back['res2a_branch1'] = 1 # 84.69
#shift_back['res2a_branch2a'] =  # x
shift_back['res2a_branch2b'] = 2
shift_back['res2b_branch2a'] = 2
shift_back['res2b_branch2b'] = 2
shift_back['res3a_branch2b'] = 2
#shift_back['res3a_branch1'] =  # x
shift_back['res3b_branch2a'] = 3
shift_back['res3b_branch2b'] = 3
shift_back['res4a_branch1'] = 3
shift_back['res4a_branch2a'] = 1 # tail weights are very sensitive 
shift_back['res4b_branch2a'] = 1
shift_back['res4b_branch2b'] = 4
shift_back['res5a_branch1'] = 3
shift_back['res5a_branch2a'] = 2
shift_back['res5a_branch2b'] = 2 # 3
shift_back['res5b_branch2a'] = 3 # 
shift_back['res5b_branch2b'] = 2 
#shift_back['fc1000'] = 0 # 1
'''



'''	
shift_back['fc1000'] = 8
shift_back['res2a_branch2b'] = 7
shift_back['res2b_branch2a'] = 7
shift_back['res2b_branch2b'] = 8
shift_back['res3a_branch1'] = 8
shift_back['res3a_branch2a'] = 8
shift_back['res3a_branch2b'] = 8
shift_back['res3b_branch2a'] = 8
shift_back['res3b_branch2b'] = 7
shift_back['res3b_branch2b'] = 8
shift_back['res4a_branch1'] = 7
shift_back['res4a_branch2a'] = 8
shift_back['res4a_branch2b'] = 8
shift_back['res4b_branch2a'] = 8
shift_back['res4b_branch2b'] = 7
shift_back['res5a_branch1'] = 8
shift_back['res5a_branch2a'] = 7
shift_back['res5a_branch2b'] = 7
shift_back['res5b_branch2a'] = 8 # **
shift_back['res5b_branch2b'] = 8 
#shift_back['conv1'] = 1 
'''
'''
shift_back = {}
layers_sorted = get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
for layer in layers_sorted:
	shift_back[layer] = 0

# (1)/3
shift_back['res2a_branch2b'] = 2
shift_back['res2b_branch2a'] = 2
shift_back['res2b_branch2b'] = 2
shift_back['res3a_branch1'] = 0
shift_back['res3a_branch2a'] = 0
shift_back['res3a_branch2b'] = 2
shift_back['res3b_branch2a'] = 2

shift_back['res3b_branch2b'] = 3
shift_back['res4a_branch1'] = 3

shift_back['res4a_branch2a'] = 2
#shift_back['res4a_branch2b'] = 0
shift_back['res4b_branch2a'] = 2
shift_back['res4b_branch2b'] = 3
shift_back['res5a_branch1'] = 1
'''



'''
shift_back['res3a_branch2b'] = 2
shift_back['res3b_branch2a'] = 2
shift_back['res3b_branch2b'] = 2
shift_back['res4a_branch1'] = 2
shift_back['res4a_branch2a'] = 2
shift_back['res4a_branch2b'] = 2
shift_back['res4b_branch2a'] = 2
shift_back['res4b_branch2b'] = 2
shift_back['res5a_branch1'] = 1
shift_back['res5a_branch2a'] = 2
shift_back['res5a_branch2b'] = 2
shift_back['res5b_branch2a'] = 2
shift_back['res5b_branch2b'] = 2
'''


path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/'
# (2)/3
path_save_q = path_save + 'resnet18_layers_shift_quant_res2b_branch2a-2sh-3bits_14May.pickle'

#shift_layers(layers_sorted, shift_back, path_ckpt, path_save)
#print('===========================')
#print('DONE! shifitng and quantization')
#print('===========================')



#net_range_normalize(path_ckpt, path_save)
'''
# =================================================================================
# Normalization check for ALEXNET 
# -------------------------------
def net_range_normalize(path_ckpt, path_save):
	#path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
	path_ckpt = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
	netparams = get_netparams(path_ckpt)
	weights, biases = netparams['weights'], netparams['biases']
	dict_shift_perc = {}
	for layer in weights:
		weights_temp = weights[layer].ravel()
		dict_shift_perc[layer] = []
		for n in range(1, 32):
			percent = ((-1/2**n < weights_temp) & (weights_temp < 1/2**n)).sum() / (len(weights_temp) + 0.)
			if percent < 0.7: # keeping at least 70% of the weights 
				break
			dict_shift_perc[layer].append((n, percent))

path_save = '/home/ahmed/projects/NN_quant/results/normalized/alexnet/'
layers = get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
path_net = 
#layers.remove('res2b_branch2a')
#layers.remove('res5b_branch2b')
#layers.remove('fc1000')

for layer in layers:
	for tuple in dict_shift_perc[layer]:
		if tuple[1] == 1.:
			continue
		netparams = get_netparams(path_ckpt)
		weights_normalized = netparams['weights']
		temp = []
		temp = weights_normalized[layer].ravel()
		temp[temp > 1/2**tuple[0]] = 1/2**tuple[0] - 1/2**32
		temp[temp < -1/2**tuple[0]] = -(1/2**tuple[0] - 1/2**32)	
		weights_normalized = quantize_weights(weights_normalized, 10)
		#with open(path_save + layer + str(tuple[0]) + '.pickle', 'w') as f:
		with open(path_save + layer + str(tuple[0]) + '.pickle', 'w') as f:
			pickle.dump([weights_normalized, biases, mean, variance, scale, offset], f)
# =================================================================================
'''
