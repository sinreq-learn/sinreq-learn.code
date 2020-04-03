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

def squeezenet_noisy(input_node, netparams, err_mean, err_stddev, train_vars):
	weights_noisy, biases_noisy, err_w, err_b = helper.add_noise(netparams['weights'], netparams['biases'], err_mean, err_stddev, train_vars)
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	err_lyr = {}
	layers_err  = {}
	data_spec = helper.get_data_spec('squeezenet')
	err_lyr['input'] = tf.get_variable(name='input_lyr_err', shape=(1, data_spec.crop_size, data_spec.crop_size, data_spec.channels), initializer=tf.random_normal_initializer(mean=err_mean[0], stddev=err_stddev[0]), trainable=train_vars[0])
	input_node_noisy = tf.add(input_node, err_lyr['input'])
	conv1 = conv(input_node_noisy, weights_noisy['conv1'], biases_noisy['conv1'], 2, 2, padding='VALID')
	err_lyr['conv1'] = tf.get_variable(name='conv1_lyr_err', shape=conv1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1'] = tf.add(conv1, err_lyr['conv1'])
	pool1 = max_pool(layers_err['conv1'], 3, 3, 2, 2, padding='VALID')
	fire2_squeeze1x1 = conv(pool1, weights_noisy['fire2_squeeze1x1'], biases_noisy['fire2_squeeze1x1'], 1, 1)
	err_lyr['fire2_squeeze1x1'] = tf.get_variable(name='fire2_squeeze1x1_lyr_err', shape=fire2_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire2_squeeze1x1'] = tf.add(fire2_squeeze1x1, err_lyr['fire2_squeeze1x1'])
	fire2_expand1x1 = conv(layers_err['fire2_squeeze1x1'], weights_noisy['fire2_expand1x1'], biases_noisy['fire2_expand1x1'], 1, 1)
	err_lyr['fire2_expand1x1'] = tf.get_variable(name='fire2_expand1x1_lyr_err', shape=fire2_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire2_expand1x1'] = tf.add(fire2_expand1x1, err_lyr['fire2_expand1x1'])
	fire2_expand3x3 = conv(layers_err['fire2_squeeze1x1'], weights_noisy['fire2_expand3x3'], biases_noisy['fire2_expand3x3'], 1, 1)
	err_lyr['fire2_expand3x3'] = tf.get_variable(name='fire2_expand3x3_lyr_err', shape=fire2_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire2_expand3x3'] = tf.add(fire2_expand3x3, err_lyr['fire2_expand3x3'])
	fire2_concat = concat([layers_err['fire2_expand1x1'], layers_err['fire2_expand3x3']], 3)
	fire3_squeeze1x1 = conv(fire2_concat, weights_noisy['fire3_squeeze1x1'], biases_noisy['fire3_squeeze1x1'], 1, 1)
	err_lyr['fire3_squeeze1x1'] = tf.get_variable(name='fire3_squeeze1x1_lyr_err', shape=fire3_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire3_squeeze1x1'] = tf.add(fire3_squeeze1x1, err_lyr['fire3_squeeze1x1'])
	fire3_expand1x1 = conv(layers_err['fire3_squeeze1x1'], weights_noisy['fire3_expand1x1'], biases_noisy['fire3_expand1x1'], 1, 1)
	err_lyr['fire3_expand1x1'] = tf.get_variable(name='fire3_expand1x1_lyr_err', shape=fire3_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire3_expand1x1'] = tf.add(fire3_expand1x1, err_lyr['fire3_expand1x1'])
	fire3_expand3x3 = conv(layers_err['fire3_squeeze1x1'], weights_noisy['fire3_expand3x3'], biases_noisy['fire3_expand3x3'], 1, 1)
	err_lyr['fire3_expand3x3'] = tf.get_variable(name='fire3_expand3x3_lyr_err', shape=fire3_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire3_expand3x3'] = tf.add(fire3_expand3x3, err_lyr['fire3_expand3x3'])
	fire3_concat = concat([layers_err['fire3_expand1x1'], layers_err['fire3_expand3x3']], 3)
	fire4_squeeze1x1 = conv(fire3_concat, weights_noisy['fire4_squeeze1x1'], biases_noisy['fire4_squeeze1x1'], 1, 1)
	err_lyr['fire4_squeeze1x1'] = tf.get_variable(name='fire4_squeeze1x1_lyr_err', shape=fire4_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire4_squeeze1x1'] = tf.add(fire4_squeeze1x1, err_lyr['fire4_squeeze1x1'])
	fire4_expand1x1 = conv(layers_err['fire4_squeeze1x1'], weights_noisy['fire4_expand1x1'], biases_noisy['fire4_expand1x1'], 1, 1)
	err_lyr['fire4_expand1x1'] = tf.get_variable(name='fire4_expand1x1_lyr_err', shape=fire4_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire4_expand1x1'] = tf.add(fire4_expand1x1, err_lyr['fire4_expand1x1'])
	fire4_expand3x3 = conv(layers_err['fire4_squeeze1x1'], weights_noisy['fire4_expand3x3'], biases_noisy['fire4_expand3x3'], 1, 1)
	err_lyr['fire4_expand3x3'] = tf.get_variable(name='fire4_expand3x3_lyr_err', shape=fire4_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire4_expand3x3'] = tf.add(fire4_expand3x3, err_lyr['fire4_expand3x3'])
	fire4_concat = concat([layers_err['fire4_expand1x1'], layers_err['fire4_expand3x3']], 3)
	pool4 = max_pool(fire4_concat, 3, 3, 2, 2, padding='VALID')
	fire5_squeeze1x1 = conv(pool4, weights_noisy['fire5_squeeze1x1'], biases_noisy['fire5_squeeze1x1'], 1, 1)
	err_lyr['fire5_squeeze1x1'] = tf.get_variable(name='fire5_squeeze1x1_lyr_err', shape=fire5_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire5_squeeze1x1'] = tf.add(fire5_squeeze1x1, err_lyr['fire5_squeeze1x1'])
	fire5_expand1x1 = conv(layers_err['fire5_squeeze1x1'], weights_noisy['fire5_expand1x1'], biases_noisy['fire5_expand1x1'], 1, 1)
	err_lyr['fire5_expand1x1'] = tf.get_variable(name='fire5_expand1x1_lyr_err', shape=fire5_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire5_expand1x1'] = tf.add(fire5_expand1x1, err_lyr['fire5_expand1x1'])
	fire5_expand3x3 = conv(layers_err['fire5_squeeze1x1'], weights_noisy['fire5_expand3x3'], biases_noisy['fire5_expand3x3'], 1, 1)
	err_lyr['fire5_expand3x3'] = tf.get_variable(name='fire5_expand3x3_lyr_err', shape=fire5_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire5_expand3x3'] = tf.add(fire5_expand3x3, err_lyr['fire5_expand3x3'])
	fire5_concat = concat([layers_err['fire5_expand1x1'], layers_err['fire5_expand3x3']], 3)
	fire6_squeeze1x1 = conv(fire5_concat, weights_noisy['fire6_squeeze1x1'], biases_noisy['fire6_squeeze1x1'], 1, 1)
	err_lyr['fire6_squeeze1x1'] = tf.get_variable(name='fire6_squeeze1x1_lyr_err', shape=fire6_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire6_squeeze1x1'] = tf.add(fire6_squeeze1x1, err_lyr['fire6_squeeze1x1'])
	fire6_expand1x1 = conv(layers_err['fire6_squeeze1x1'], weights_noisy['fire6_expand1x1'], biases_noisy['fire6_expand1x1'], 1, 1)
	err_lyr['fire6_expand1x1'] = tf.get_variable(name='fire6_expand1x1_lyr_err', shape=fire6_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire6_expand1x1'] = tf.add(fire6_expand1x1, err_lyr['fire6_expand1x1'])
	fire6_expand3x3 = conv(layers_err['fire6_squeeze1x1'], weights_noisy['fire6_expand3x3'], biases_noisy['fire6_expand3x3'], 1, 1)
	err_lyr['fire6_expand3x3'] = tf.get_variable(name='fire6_expand3x3_lyr_err', shape=fire6_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire6_expand3x3'] = tf.add(fire6_expand3x3, err_lyr['fire6_expand3x3'])
	fire6_concat = concat([layers_err['fire6_expand1x1'], layers_err['fire6_expand3x3']], 3)
	fire7_squeeze1x1 = conv(fire6_concat, weights_noisy['fire7_squeeze1x1'], biases_noisy['fire7_squeeze1x1'], 1, 1)
	err_lyr['fire7_squeeze1x1'] = tf.get_variable(name='fire7_squeeze1x1_lyr_err', shape=fire7_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire7_squeeze1x1'] = tf.add(fire7_squeeze1x1, err_lyr['fire7_squeeze1x1'])
	fire7_expand1x1 = conv(layers_err['fire7_squeeze1x1'], weights_noisy['fire7_expand1x1'], biases_noisy['fire7_expand1x1'], 1, 1)
	err_lyr['fire7_expand1x1'] = tf.get_variable(name='fire7_expand1x1_lyr_err', shape=fire7_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire7_expand1x1'] = tf.add(fire7_expand1x1, err_lyr['fire7_expand1x1'])
	fire7_expand3x3 = conv(layers_err['fire7_squeeze1x1'], weights_noisy['fire7_expand3x3'], biases_noisy['fire7_expand3x3'], 1, 1)
	err_lyr['fire7_expand3x3'] = tf.get_variable(name='fire7_expand3x3_lyr_err', shape=fire7_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire7_expand3x3'] = tf.add(fire7_expand3x3, err_lyr['fire7_expand3x3'])
	fire7_concat = concat([layers_err['fire7_expand1x1'], layers_err['fire7_expand3x3']], 3)
	fire8_squeeze1x1 = conv(fire7_concat, weights_noisy['fire8_squeeze1x1'], biases_noisy['fire8_squeeze1x1'], 1, 1)
	err_lyr['fire8_squeeze1x1'] = tf.get_variable(name='fire8_squeeze1x1_lyr_err', shape=fire8_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire8_squeeze1x1'] = tf.add(fire8_squeeze1x1, err_lyr['fire8_squeeze1x1'])
	fire8_expand1x1 = conv(layers_err['fire8_squeeze1x1'], weights_noisy['fire8_expand1x1'], biases_noisy['fire8_expand1x1'], 1, 1)
	err_lyr['fire8_expand1x1'] = tf.get_variable(name='fire8_expand1x1_lyr_err', shape=fire8_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire8_expand1x1'] = tf.add(fire8_expand1x1, err_lyr['fire8_expand1x1'])
	fire8_expand3x3 = conv(layers_err['fire8_squeeze1x1'], weights_noisy['fire8_expand3x3'], biases_noisy['fire8_expand3x3'], 1, 1)
	err_lyr['fire8_expand3x3'] = tf.get_variable(name='fire8_expand3x3_lyr_err', shape=fire8_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire8_expand3x3'] = tf.add(fire8_expand3x3, err_lyr['fire8_expand3x3'])
	fire8_concat = concat([layers_err['fire8_expand1x1'], layers_err['fire8_expand3x3']], 3)
	pool8 = max_pool(fire8_concat, 3, 3, 2, 2, padding='VALID')
	fire9_squeeze1x1 = conv(pool8, weights_noisy['fire9_squeeze1x1'], biases_noisy['fire9_squeeze1x1'], 1, 1)
	err_lyr['fire9_squeeze1x1'] = tf.get_variable(name='fire9_squeeze1x1_lyr_err', shape=fire9_squeeze1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire9_squeeze1x1'] = tf.add(fire9_squeeze1x1, err_lyr['fire9_squeeze1x1'])
	fire9_expand1x1 = conv(layers_err['fire9_squeeze1x1'], weights_noisy['fire9_expand1x1'], biases_noisy['fire9_expand1x1'], 1, 1)
	err_lyr['fire9_expand1x1'] = tf.get_variable(name='fire9_expand1x1_lyr_err', shape=fire9_expand1x1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire9_expand1x1'] = tf.add(fire9_expand1x1, err_lyr['fire9_expand1x1'])
	fire9_expand3x3 = conv(layers_err['fire9_squeeze1x1'], weights_noisy['fire9_expand3x3'], biases_noisy['fire9_expand3x3'], 1, 1)
	err_lyr['fire9_expand3x3'] = tf.get_variable(name='fire9_expand3x3_lyr_err', shape=fire9_expand3x3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['fire9_expand3x3'] = tf.add(fire9_expand3x3, err_lyr['fire9_expand3x3'])
	fire9_concat = concat([layers_err['fire9_expand1x1'], layers_err['fire9_expand3x3']], 3)
	conv10 = conv(fire9_concat, weights_noisy['conv10'], biases_noisy['conv10'], 1, 1)
	err_lyr['conv10'] = tf.get_variable(name='conv10_lyr_err', shape=conv10.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv10'] = tf.add(conv10, err_lyr['conv10'])
	pool10 = avg_pool(layers_err['conv10'], 13, 13, 1, 1, padding='VALID')
	return pool10, err_w, err_b, err_lyr

def squeezenet(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']

	data_spec = get_data_spec('squeezenet')
	
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 2, 2, padding='VALID')
	pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')

	fire2_squeeze1x1 = conv(pool1, weights['fire2_squeeze1x1'], biases['fire2_squeeze1x1'], 1, 1)

	fire2_expand1x1 = conv(fire2_squeeze1x1, weights['fire2_expand1x1'], biases['fire2_expand1x1'], 1, 1)
	fire2_expand3x3 = conv(fire2_squeeze1x1, weights['fire2_expand3x3'], biases['fire2_expand3x3'], 1, 1)
	fire2_concat = concat([fire2_expand1x1, fire2_expand3x3], 3)
	
	fire3_squeeze1x1 = conv(fire2_concat, weights['fire3_squeeze1x1'], biases['fire3_squeeze1x1'], 1, 1)
	
	fire3_expand1x1 = conv(fire3_squeeze1x1, weights['fire3_expand1x1'], biases['fire3_expand1x1'], 1, 1)
	fire3_expand3x3 = conv(fire3_squeeze1x1, weights['fire3_expand3x3'], biases['fire3_expand3x3'], 1, 1)
	fire3_concat = concat([fire3_expand1x1, fire3_expand3x3], 3)
	
	fire4_squeeze1x1 = conv(fire3_concat, weights['fire4_squeeze1x1'], biases['fire4_squeeze1x1'], 1, 1)
	
	fire4_expand1x1 = conv(fire4_squeeze1x1, weights['fire4_expand1x1'], biases['fire4_expand1x1'], 1, 1)
	fire4_expand3x3 = conv(fire4_squeeze1x1, weights['fire4_expand3x3'], biases['fire4_expand3x3'], 1, 1)
	fire4_concat = concat([fire4_expand1x1, fire4_expand3x3], 3)

	pool4 = max_pool(fire4_concat, 3, 3, 2, 2, padding='VALID')
	fire5_squeeze1x1 = conv(pool4, weights['fire5_squeeze1x1'], biases['fire5_squeeze1x1'], 1, 1)
	
	fire5_expand1x1 = conv(fire5_squeeze1x1, weights['fire5_expand1x1'], biases['fire5_expand1x1'], 1, 1)
	fire5_expand3x3 = conv(fire5_squeeze1x1, weights['fire5_expand3x3'], biases['fire5_expand3x3'], 1, 1)
	fire5_concat = concat([fire5_expand1x1, fire5_expand3x3], 3)

	fire6_squeeze1x1 = conv(fire5_concat, weights['fire6_squeeze1x1'], biases['fire6_squeeze1x1'], 1, 1)
	
	fire6_expand1x1 = conv(fire6_squeeze1x1, weights['fire6_expand1x1'], biases['fire6_expand1x1'], 1, 1)
	fire6_expand3x3 = conv(fire6_squeeze1x1, weights['fire6_expand3x3'], biases['fire6_expand3x3'], 1, 1)
	fire6_concat = concat([fire6_expand1x1, fire6_expand3x3], 3)

	fire7_squeeze1x1 = conv(fire6_concat, weights['fire7_squeeze1x1'], biases['fire7_squeeze1x1'], 1, 1)

	fire7_expand1x1 = conv(fire7_squeeze1x1, weights['fire7_expand1x1'], biases['fire7_expand1x1'], 1, 1)
	fire7_expand3x3 = conv(fire7_squeeze1x1, weights['fire7_expand3x3'], biases['fire7_expand3x3'], 1, 1)
	fire7_concat = concat([fire7_expand1x1, fire7_expand3x3], 3)

	fire8_squeeze1x1 = conv(fire7_concat, weights['fire8_squeeze1x1'], biases['fire8_squeeze1x1'], 1, 1)

	fire8_expand1x1 = conv(fire8_squeeze1x1, weights['fire8_expand1x1'], biases['fire8_expand1x1'], 1, 1)
	fire8_expand3x3 = conv(fire8_squeeze1x1, weights['fire8_expand3x3'], biases['fire8_expand3x3'], 1, 1)
	fire8_concat = concat([fire8_expand1x1, fire8_expand3x3], 3)
	pool8 = max_pool(fire8_concat, 3, 3, 2, 2, padding='VALID')
	
	fire9_squeeze1x1 = conv(pool8, weights['fire9_squeeze1x1'], biases['fire9_squeeze1x1'], 1, 1)
	
	fire9_expand1x1 = conv(fire9_squeeze1x1, weights['fire9_expand1x1'], biases['fire9_expand1x1'], 1, 1)
	fire9_expand3x3 = conv(fire9_squeeze1x1, weights['fire9_expand3x3'], biases['fire9_expand3x3'], 1, 1)
	fire9_concat = concat([fire9_expand1x1, fire9_expand3x3], 3)

	conv10 = conv(fire9_concat, weights['conv10'], biases['conv10'], 1, 1)
	pool10 = avg_pool(conv10, 13, 13, 1, 1, padding='VALID')
	return pool10

def squeezenet_q(input_node, netparams, qbits=[]):
	#qbits = [3] * 26
	# first and last layers are left in full precision 
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']

	data_spec = get_data_spec('squeezenet')
	
	conv1 = conv(input_node, weights['conv1'], biases['conv1'], 2, 2, padding='VALID')
	pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')

	# layer 1
	k = qbits[1]
	weights_fire2_squeeze1x1_q = quantize_weights(weights['fire2_squeeze1x1'], k)
	fire2_squeeze1x1 = conv(pool1, weights_fire2_squeeze1x1_q, biases['fire2_squeeze1x1'], 1, 1)

	
	# layer 2
	k = qbits[2]
	weights_fire2_expand1x1_q = quantize_weights(weights['fire2_expand1x1'], k)
	fire2_expand1x1 = conv(fire2_squeeze1x1, weights_fire2_expand1x1_q, biases['fire2_expand1x1'], 1, 1)
	# layer 3
	k = qbits[3]
	weights_fire2_expand3x3_q = quantize_weights(weights['fire2_expand3x3'], k)
	fire2_expand3x3 = conv(fire2_squeeze1x1, weights_fire2_expand3x3_q, biases['fire2_expand3x3'], 1, 1)
	
	fire2_concat = concat([fire2_expand1x1, fire2_expand3x3], 3)
	
	# layer 4
	k = qbits[4]
	weights_fire3_squeeze1x1_q = quantize_weights(weights['fire3_squeeze1x1'], k)
	fire3_squeeze1x1 = conv(fire2_concat, weights_fire3_squeeze1x1_q, biases['fire3_squeeze1x1'], 1, 1)
	
	# layer 5
	k = qbits[5]
	weights_fire3_expand1x1_q = quantize_weights(weights['fire3_expand1x1'], k)
	fire3_expand1x1 = conv(fire3_squeeze1x1, weights_fire3_expand1x1_q, biases['fire3_expand1x1'], 1, 1)
	# layer 6
	k = qbits[6]
	weights_fire3_expand3x3_q = quantize_weights(weights['fire3_expand3x3'], k)
	fire3_expand3x3 = conv(fire3_squeeze1x1, weights_fire3_expand3x3_q, biases['fire3_expand3x3'], 1, 1)
	fire3_concat = concat([fire3_expand1x1, fire3_expand3x3], 3)
	
	# layer 7
	k = qbits[7]
	weights_fire4_squeeze1x1_q = quantize_weights(weights['fire4_squeeze1x1'], k)
	fire4_squeeze1x1 = conv(fire3_concat, weights_fire4_squeeze1x1_q, biases['fire4_squeeze1x1'], 1, 1)
	
	# layer 8
	k = qbits[8]
	weights_fire4_expand1x1_q = quantize_weights(weights['fire4_expand1x1'], k)
	fire4_expand1x1 = conv(fire4_squeeze1x1, weights_fire4_expand1x1_q, biases['fire4_expand1x1'], 1, 1)
	# layer 9
	k = qbits[9]
	weights_fire4_expand3x3_q = quantize_weights(weights['fire4_expand3x3'], k)
	fire4_expand3x3 = conv(fire4_squeeze1x1, weights_fire4_expand3x3_q, biases['fire4_expand3x3'], 1, 1)
	fire4_concat = concat([fire4_expand1x1, fire4_expand3x3], 3)
	pool4 = max_pool(fire4_concat, 3, 3, 2, 2, padding='VALID')
	
	# layer 10
	k = qbits[10]
	weights_fire5_squeeze1x1_q = quantize_weights(weights['fire5_squeeze1x1'], k)
	fire5_squeeze1x1 = conv(pool4, weights_fire5_squeeze1x1_q, biases['fire5_squeeze1x1'], 1, 1)
	
	# layer 11
	k = qbits[11]
	weights_fire5_expand1x1_q = quantize_weights(weights['fire5_expand1x1'], k)
	fire5_expand1x1 = conv(fire5_squeeze1x1, weights_fire5_expand1x1_q, biases['fire5_expand1x1'], 1, 1)
	# layer 12
	k = qbits[12]
	weights_fire5_expand3x3_q = quantize_weights(weights['fire5_expand3x3'], k)
	fire5_expand3x3 = conv(fire5_squeeze1x1, weights_fire5_expand3x3_q, biases['fire5_expand3x3'], 1, 1)
	fire5_concat = concat([fire5_expand1x1, fire5_expand3x3], 3)

	# layer 13
	k = qbits[13]
	weights_fire6_squeeze1x1_q = quantize_weights(weights['fire6_squeeze1x1'], k)
	fire6_squeeze1x1 = conv(fire5_concat, weights_fire6_squeeze1x1_q, biases['fire6_squeeze1x1'], 1, 1)
	
	# layer 14
	k = qbits[14]
	weights_fire6_expand1x1_q = quantize_weights(weights['fire6_expand1x1'], k)
	fire6_expand1x1 = conv(fire6_squeeze1x1, weights_fire6_expand1x1_q, biases['fire6_expand1x1'], 1, 1)
	# layer 15
	k = qbits[15]
	weights_fire6_expand3x3_q = quantize_weights(weights['fire6_expand3x3'], k)
	fire6_expand3x3 = conv(fire6_squeeze1x1, weights_fire6_expand3x3_q, biases['fire6_expand3x3'], 1, 1)
	fire6_concat = concat([fire6_expand1x1, fire6_expand3x3], 3)

	# layer 16
	k = qbits[16]
	weights_fire7_squeeze1x1_q = quantize_weights(weights['fire7_squeeze1x1'], k)
	fire7_squeeze1x1 = conv(fire6_concat, weights_fire7_squeeze1x1_q, biases['fire7_squeeze1x1'], 1, 1)

	# layer 17
	k = qbits[17]
	weights_fire7_expand1x1_q = quantize_weights(weights['fire7_expand1x1'], k)
	fire7_expand1x1 = conv(fire7_squeeze1x1, weights_fire7_expand1x1_q, biases['fire7_expand1x1'], 1, 1)
	# layer 18
	k = qbits[18]
	weights_fire7_expand3x3_q = quantize_weights(weights['fire7_expand3x3'], k)
	fire7_expand3x3 = conv(fire7_squeeze1x1, weights_fire7_expand3x3_q, biases['fire7_expand3x3'], 1, 1)
	fire7_concat = concat([fire7_expand1x1, fire7_expand3x3], 3)

	# layer 19
	k = qbits[19]
	weights_fire8_squeeze1x1_q = quantize_weights(weights['fire8_squeeze1x1'], k)
	fire8_squeeze1x1 = conv(fire7_concat, weights_fire8_squeeze1x1_q, biases['fire8_squeeze1x1'], 1, 1)

	# layer 20
	k = qbits[20]
	weights_fire8_expand1x1_q = quantize_weights(weights['fire8_expand1x1'], k)
	fire8_expand1x1 = conv(fire8_squeeze1x1, weights_fire8_expand1x1_q, biases['fire8_expand1x1'], 1, 1)
	# layer 21
	k = qbits[21]
	weights_fire8_expand3x3_q = quantize_weights(weights['fire8_expand3x3'], k)
	fire8_expand3x3 = conv(fire8_squeeze1x1, weights_fire8_expand3x3_q, biases['fire8_expand3x3'], 1, 1)
	fire8_concat = concat([fire8_expand1x1, fire8_expand3x3], 3)
	pool8 = max_pool(fire8_concat, 3, 3, 2, 2, padding='VALID')
	
	# layer 22
	k = qbits[22]
	weights_fire9_squeeze1x1_q = quantize_weights(weights['fire9_squeeze1x1'], k)
	fire9_squeeze1x1 = conv(pool8, weights_fire9_squeeze1x1_q, biases['fire9_squeeze1x1'], 1, 1)
	
	# layer 23
	k = qbits[23]
	weights_fire9_expand1x1_q = quantize_weights(weights['fire9_expand1x1'], k)
	fire9_expand1x1 = conv(fire9_squeeze1x1, weights_fire9_expand1x1_q, biases['fire9_expand1x1'], 1, 1)
	# layer 24
	k = qbits[24]
	weights_fire9_expand3x3_q = quantize_weights(weights['fire9_expand3x3'], k)
	fire9_expand3x3 = conv(fire9_squeeze1x1, weights_fire9_expand3x3_q, biases['fire9_expand3x3'], 1, 1)
	fire9_concat = concat([fire9_expand1x1, fire9_expand3x3], 3)

	conv10 = conv(fire9_concat, weights['conv10'], biases['conv10'], 1, 1)
	pool10 = avg_pool(conv10, 13, 13, 1, 1, padding='VALID')
	return pool10

