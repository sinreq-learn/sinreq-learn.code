import tensorflow as tf
#import helper
#from helper import *
from .helper import *

def quantize_acti(x, k):
    mini = tf.reduce_min(x)
    maxi = tf.reduce_max(x)
    x = (x - mini)/(maxi - mini)
    G = tf.get_default_graph()
    n = float(2**k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

def quantize_weights(x, k):
    #mini = tf.reduce_min(x)
    #maxi = tf.reduce_max(x)
    #x = (x - mini)/(maxi - mini)
    G = tf.get_default_graph()
    n = float(2**k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

def lenet_noisy(input_node, netparams, err_mean, err_stddev, train_vars):
	weights_noisy, biases_noisy, err_w, err_b = helper.add_noise(netparams['weights'], netparams['biases'], err_mean, err_stddev, train_vars)
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	err_lyr = {}
	layers_err  = {}
	data_spec = helper.get_data_spec('lenet')
	err_lyr['input'] = tf.get_variable(name='input_lyr_err', shape=(1, data_spec.crop_size, data_spec.crop_size, data_spec.channels), initializer=tf.random_normal_initializer(mean=err_mean[0], stddev=err_stddev[0]), trainable=train_vars[0])
	input_node_noisy = tf.add(input_node, err_lyr['input'])
	conv1 = conv(input_node_noisy, weights_noisy['conv1'], biases_noisy['conv1'], 1, 1, padding='VALID', relu=False)
	err_lyr['conv1'] = tf.get_variable(name='conv1_lyr_err', shape=conv1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1'] = tf.add(conv1, err_lyr['conv1'])
	pool1 = max_pool(layers_err['conv1'], 2, 2, 2, 2)
	conv2 = conv(pool1, weights_noisy['conv2'], biases_noisy['conv2'], 1, 1, padding='VALID', relu=False)
	err_lyr['conv2'] = tf.get_variable(name='conv2_lyr_err', shape=conv2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv2'] = tf.add(conv2, err_lyr['conv2'])
	pool2 = max_pool(layers_err['conv2'], 2, 2, 2, 2)
	ip1 = fc(pool2, weights_noisy['ip1'], biases_noisy['ip1'])
	err_lyr['ip1'] = tf.get_variable(name='ip1_lyr_err', shape=ip1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['ip1'] = tf.add(ip1, err_lyr['ip1'])
	ip2 = fc(ip1, weights_noisy['ip2'], biases_noisy['ip2'], relu=False)
	err_lyr['ip2'] = tf.get_variable(name='ip2_lyr_err', shape=ip2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['ip2'] = tf.add(ip2, err_lyr['ip2'])
	return layers_err['ip2'], err_w, err_b, err_lyr

def lenet(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	data_spec = get_data_spec('lenet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 1, 1, padding='VALID', relu=False)
	#conv1 = conv(input_node, weights['conv1'], biases['conv1'], 1, 1, padding='VALID', relu=False, biased= False)
	pool1 = max_pool(conv1, 2, 2, 2, 2)
	conv2 = conv(pool1, weights['conv2'], biases['conv2'], 1, 1, padding='VALID', relu=False)
	pool2 = max_pool(conv2, 2, 2, 2, 2)
	ip1 = fc(pool2, weights['ip1'], biases['ip1'])
	ip2 = fc(ip1, weights['ip2'], biases['ip2'], relu=False)
	return ip2, input_node  

def lenet_quantized(input_node, netparams, qbits):
	#qbits = [4,4,4,4,4]
	num_layers = 4
	layer_quantize = [1] * num_layers

	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	data_spec = get_data_spec('lenet')

	k = qbits[0]
	if layer_quantize[0] == 0:
		weights_conv1_q = weights['conv1']
	else:
		weights_conv1_q = quantize_weights(weights['conv1'], k)
	conv1 = conv(input_node, weights_conv1_q, biases['conv1'], 1, 1, padding='VALID', relu=False)
	pool1 = max_pool(conv1, 2, 2, 2, 2)


	k = qbits[1]
	if layer_quantize[1] == 0:
		weights_conv2_q = weights['conv2']
	else:
		weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, padding='VALID', relu=False)
	pool2 = max_pool(conv2, 2, 2, 2, 2)


	k = qbits[2]
	if layer_quantize[2] == 0:
		weights_ip1_q = weights['ip1']
	else:
		weights_ip1_q = quantize_weights(weights['ip1'], k)
	ip1 = fc(pool2, weights_ip1_q, biases['ip1'])
	

	k = qbits[3]
	if layer_quantize[3] == 0:
		weights_ip2_q = weights['ip2']
	else:
		weights_ip2_q = quantize_weights(weights['ip2'], k)
	ip2 = fc(ip1, weights_ip2_q, biases['ip2'], relu=False)

	return ip2, weights_conv1_q

def lenet_q_RL(input_node, netparams, qbits, layer_idx):
	#qbits = [4,4,4,4]

	num_layers = 8
	layer_quantize = [0] * num_layers
	layer_quantize[0:layer_idx] = [1] * (layer_idx)

	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	data_spec = get_data_spec('lenet')


	k = qbits[0]
	if layer_quantize[1] == 0:
		weights_conv1_q = weights['conv1']
	else:
		weights_conv1_q = quantize_weights(weights['conv1'], k)
	conv1 = conv(input_node, weights_conv1_q, biases['conv1'], 1, 1, padding='VALID', relu=False)
	pool1 = max_pool(conv1, 2, 2, 2, 2)


	k = qbits[1]
	if layer_quantize[1] == 0:
		weights_conv2_q = weights['conv2']
	else:
		weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, padding='VALID', relu=False)
	pool2 = max_pool(conv2, 2, 2, 2, 2)


	k = qbits[2]
	if layer_quantize[2] == 0:
		weights_ip1_q = weights['ip1']
	else:
		weights_ip1_q = quantize_weights(weights['ip1'], k)
	ip1 = fc(pool2, weights_ip1_q, biases['ip1'])
	

	k = qbits[3]
	if layer_quantize[3] == 0:
		weights_ip2_q = weights['ip2']
	else:
		weights_ip2_q = quantize_weights(weights['ip2'], k)
	ip2 = fc(ip1, weights_ip2_q, biases['ip2'], relu=False)

	return ip2
