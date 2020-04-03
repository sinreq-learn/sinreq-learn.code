import tensorflow as tf
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


def alexnet_noisy(input_node, netparams, err_mean, err_stddev, train_vars):
	weights_noisy, biases_noisy, err_w, err_b = add_noise(netparams['weights'], netparams['biases'], err_mean, err_stddev, train_vars)
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	err_lyr = {}
	layers_err  = {}
	data_spec = get_data_spec('alexnet')
	err_lyr['input'] = tf.get_variable(name='input_lyr_err', shape=(1, data_spec.crop_size, data_spec.crop_size, data_spec.channels), initializer=tf.random_normal_initializer(mean=err_mean[0], stddev=err_stddev[0]), trainable=train_vars[0])
	input_node_noisy = tf.add(input_node, err_lyr['input'])
	conv1 = conv(input_node_noisy, weights_noisy['conv1'], biases_noisy['conv1'], 4, 4, padding='VALID')
	err_lyr['conv1'] = tf.get_variable(name='conv1_lyr_err', shape=conv1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1'] = tf.add(conv1, err_lyr['conv1'])
	norm1 = lrn(layers_err['conv1'], 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	conv2 = conv(pool1, weights_noisy['conv2'], biases_noisy['conv2'], 1, 1, group=2)
	err_lyr['conv2'] = tf.get_variable(name='conv2_lyr_err', shape=conv2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv2'] = tf.add(conv2, err_lyr['conv2'])
	norm2 = lrn(layers_err['conv2'], 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights_noisy['conv3'], biases_noisy['conv3'], 1, 1)
	err_lyr['conv3'] = tf.get_variable(name='conv3_lyr_err', shape=conv3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv3'] = tf.add(conv3, err_lyr['conv3'])
	conv4 = conv(layers_err['conv3'], weights_noisy['conv4'], biases_noisy['conv4'], 1, 1, group=2)
	err_lyr['conv4'] = tf.get_variable(name='conv4_lyr_err', shape=conv4.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv4'] = tf.add(conv4, err_lyr['conv4'])
	conv5 = conv(layers_err['conv4'], weights_noisy['conv5'], biases_noisy['conv5'], 1, 1, group=2)
	err_lyr['conv5'] = tf.get_variable(name='conv5_lyr_err', shape=conv5.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv5'] = tf.add(conv5, err_lyr['conv5'])
	pool5 = max_pool(layers_err['conv5'], 3, 3, 2, 2, padding='VALID')
	fc6 = fc(pool5, weights_noisy['fc6'], biases_noisy['fc6'])
	err_lyr['fc6'] = tf.get_variable(name='fc6_lyr_err', shape=fc6.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fc6'] = tf.add(fc6, err_lyr['fc6'])
	fc7 = fc(fc6, weights_noisy['fc7'], biases_noisy['fc7'])
	err_lyr['fc7'] = tf.get_variable(name='fc7_lyr_err', shape=fc7.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fc7'] = tf.add(fc7, err_lyr['fc7'])
	fc8 = fc(fc7, weights_noisy['fc8'], biases_noisy['fc8'], relu=False)
	err_lyr['fc8'] = tf.get_variable(name='fc8_lyr_err', shape=fc8.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fc8'] = tf.add(fc8, err_lyr['fc8'])
	return layers_err['fc8'], err_w, err_b, err_lyr

"""
def alexnet_spoilded(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 5, 5, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	conv2 = conv(pool1, weights['conv2'], biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights['conv3'], biases['conv3'], 1, 1)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8
"""

def alexnet(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	conv2 = conv(pool1, weights['conv2'], biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights['conv3'], biases['conv3'], 1, 1)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8

def svhn_net(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('svhn_net')
	conv1 = conv(input_node, weights['hidden1'], biases['hidden1'], 4, 4, padding='VALID', relu=False)
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 2, 2, 2, 2, padding='VALID')

	conv2 = conv(pool1, weights['hidden2'], biases['hidden2'], 5, 5, group=2, relu=True)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 2, 2, 1, 1, padding='VALID')

	conv3 = conv(pool2, weights['hidden3'], biases['hidden3'], 5, 5, relu=True)
	norm3 = lrn(conv3, 2, 1.99999994948e-05, 0.75)
	pool3 = max_pool(norm3, 2, 2, 2, 2, padding='VALID')

	conv4 = conv(pool3, weights['hidden4'], biases['hidden4'], 5, 5, relu=True)
	norm4 = lrn(conv4, 2, 1.99999994948e-05, 0.75)
	pool4 = max_pool(norm4, 2, 2, 1, 1, padding='VALID')

	conv5 = conv(pool4, weights['hidden5'], biases['hidden5'], 5, 5, relu=True)
	norm5 = lrn(conv5, 2, 1.99999994948e-05, 0.75)
	pool5 = max_pool(norm5, 2, 2, 2, 2, padding='VALID')

	conv6 = conv(pool5, weights['hidden6'], biases['hidden6'], 5, 5, relu=True)
	norm6 = lrn(conv6, 2, 1.99999994948e-05, 0.75)
	pool6 = max_pool(norm6, 2, 2, 1, 1, padding='VALID')

	conv7 = conv(pool6, weights['hidden7'], biases['hidden7'], 5, 5, relu=True)
	norm7 = lrn(conv7, 2, 1.99999994948e-05, 0.75)
	pool7 = max_pool(norm7, 2, 2, 2, 2, padding='VALID')

	conv8 = conv(pool7, weights['hidden8'], biases['hidden8'], 5, 5, relu=True)
	norm8 = lrn(conv8, 2, 1.99999994948e-05, 0.75)
	pool8 = max_pool(norm8, 2, 2, 1, 1, padding='VALID')

	flatten = tf.reshape(pool8, [-1, 4 * 4 * 192])

	hidden9 = fc(flatten, weights['hidden9'], biases['hidden9'])
	
	hidden10 = fc(hidden9, weights['hidden10'], biases['hidden10'])
	
	length = fc(hidden10, weights['length'], biases['length'])

	digit1 = fc(hidden10, weights['digit1'], biases['digit1'])

	digit2 = fc(hidden10, weights['digit2'], biases['digit2'])

	digit3 = fc(hidden10, weights['digit3'], biases['digit3'])

	digit4 = fc(hidden10, weights['digit4'], biases['digit4'])

	digit5 = fc(hidden10, weights['digit5'], biases['digit5'])

	length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
	return length_logits, digits_logits

def alexnet_q(input_node, netparams, qbits):
	# qbits = [16, 8, 8, 4, 8, 8, 8, 16]

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[1]
	weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[2]
	weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	k = qbits[3]
	weights_conv4_q = quantize_weights(weights['conv4'], k)
	conv4 = conv(conv3, weights_conv4_q, biases['conv4'], 1, 1, group=2)

	k = qbits[4]
	weights_conv5_q = quantize_weights(weights['conv5'], k)
	conv5 = conv(conv4, weights_conv5_q, biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[5]
	weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights_fc6_q, biases['fc6'])

	k = qbits[6]
	weights_fc7_q = quantize_weights(weights['fc7'], k)	
	fc7 = fc(fc6, weights_fc7_q, biases['fc7'])

	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8

def alexnet_q_1(input_node, netparams):
	# qbits = [16, 8, 8, 4, 8, 8, 8, 16]

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	k = 8
	weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)

	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	k = 8
	weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	#k = 4
	#weights_conv4_q = quantize_weights(weights['conv4'], k)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)

	k = 8
	weights_conv5_q = quantize_weights(weights['conv5'], k)
	conv5 = conv(conv4, weights_conv5_q, biases['conv5'], 1, 1, group=2)

	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	k = 8
	weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights_fc6_q, biases['fc6'])

	k = 8
	weights_fc7_q = quantize_weights(weights['fc7'], k)	
	fc7 = fc(fc6, weights_fc7_q, biases['fc7'])

	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8, conv4

def alexnet_q_sin2(input_node, netparams, qbits):
	# qbits = [16, 8, 8, 4, 8, 8, 8, 16]

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[1]
	weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights['conv2'], biases['conv2'], 1, 1, group=2)

	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	k = qbits[2]
	weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights['conv3'], biases['conv3'], 1, 1)

	k = qbits[3]
	weights_conv4_q = quantize_weights(weights['conv4'], k)
	q_diff = tf.subtract(weights_conv4_q, weights['conv4'])
	#q_diff_cost = tf.nn.l2_loss(q_diff)
	q_diff_cost = tf.reduce_mean(q_diff)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)

	k = qbits[4]
	weights_conv5_q = quantize_weights(weights['conv5'], k)
	conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)

	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	k = qbits[5]
	weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])

	k = qbits[6]
	weights_fc7_q = quantize_weights(weights['fc7'], k)	
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])

	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8, weights['conv4']

def alexnet_q_RL_v0(input_node, netparams, qbits, layer_idx):
	# qbits = [16, 8, 8, 4, 8, 8, 8, 16]

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')

	""" first layer is kept in full precision """
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[1]
	if layer_idx == 1:
		weights_conv2_q = weights['conv2']
	else:
		weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[2]
	if layer_idx == 2:
		weights_conv3_q = weights['conv3']
	else:
		weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	k = qbits[3]
	if layer_idx == 3:
		weights_conv4_q = weights['conv4']
	else:
		weights_conv4_q = quantize_weights(weights['conv4'], k)
	conv4 = conv(conv3, weights_conv4_q, biases['conv4'], 1, 1, group=2)

	k = qbits[4]
	if layer_idx == 4:
		weights_conv5_q = weights['conv5']
	else:
		weights_conv5_q = quantize_weights(weights['conv5'], k)
	conv5 = conv(conv4, weights_conv5_q, biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[5]
	if layer_idx == 5:
		weights_fc6_q = weights['fc6']
	else:
		weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights_fc6_q, biases['fc6'])

	k = qbits[6]
	if layer_idx == 6:
		weights_fc7_q = weights['fc7']
	else:
		weights_fc7_q = quantize_weights(weights['fc7'], k)
	fc7 = fc(fc6, weights_fc7_q, biases['fc7'])

	""" last layer is kept in full precision """
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8, conv4

def alexnet_q_RL(input_node, netparams, qbits, layer_idx):
	# qbits = [16, 8, 8, 4, 8, 8, 8, 16]
	num_layers = 8
	layer_quantize = [0] * num_layers
	layer_quantize[0:layer_idx] = [1] * (layer_idx)

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')

	""" first layer is kept in full precision """
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[1]
	if layer_quantize[1] == 0:
		weights_conv2_q = weights['conv2']
	else:
		weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[2]
	if layer_quantize[2] == 0:
		weights_conv3_q = weights['conv3']
	else:
		weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	k = qbits[3]
	if layer_quantize[3] == 0:
		weights_conv4_q = weights['conv4']
	else:
		weights_conv4_q = quantize_weights(weights['conv4'], k)
	conv4 = conv(conv3, weights_conv4_q, biases['conv4'], 1, 1, group=2)

	k = qbits[4]
	if layer_quantize[4] == 0:
		weights_conv5_q = weights['conv5']
	else:
		weights_conv5_q = quantize_weights(weights['conv5'], k)
	conv5 = conv(conv4, weights_conv5_q, biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	
	k = qbits[5]
	if layer_quantize[5] == 0:
		weights_fc6_q = weights['fc6']
	else:
		weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights_fc6_q, biases['fc6'])

	k = qbits[6]
	if layer_quantize[6] == 0:
		weights_fc7_q = weights['fc7']
	else:
		weights_fc7_q = quantize_weights(weights['fc7'], k)
	fc7 = fc(fc6, weights_fc7_q, biases['fc7'])

	""" last layer is kept in full precision """
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8, conv4


def alexnet_conv1_conv3(input_node, netparams):
	k = 4

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)

	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	#conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	#conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	#pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	#fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	#fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	#fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return conv3, weights_conv2_q

def alexnet_conv1_conv3(input_node, netparams):
	k = 6

	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	weights_conv2_q = quantize_weights(weights['conv2'], k)
	conv2 = conv(pool1, weights_conv2_q, biases['conv2'], 1, 1, group=2)

	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	weights_conv3_q = quantize_weights(weights['conv3'], k)
	conv3 = conv(pool2, weights_conv3_q, biases['conv3'], 1, 1)

	#conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	#conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	#pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	#fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	#fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	#fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return conv3, weights_conv2_q


def alexnet_shift(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	conv2 = conv(pool1, weights['conv2'], biases['conv2'], 1, 1, group=2)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights['conv3'], biases['conv3'], 1, 1)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8
'''	
def alexnet_noisy_layer(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('alexnet')
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 4, 4, padding='VALID')
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')
	
	err_w_conv2 = tf.get_variable(name='err_w_conv2', shape=weights['conv2'].shape, initializer=tf.zeros_initializer, trainable=True)
	weights_noisy_conv2 = tf.add(err_w_conv2, weights['conv2'])
	conv2 = conv(pool1, weights_noisy_conv2, biases['conv2'], 1, 1, group=2)
	
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights['conv3'], biases['conv3'], 1, 1)
	conv4 = conv(conv3, weights['conv4'], biases['conv4'], 1, 1, group=2)
	conv5 = conv(conv4, weights['conv5'], biases['conv5'], 1, 1, group=2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8, err_w_conv2
'''
