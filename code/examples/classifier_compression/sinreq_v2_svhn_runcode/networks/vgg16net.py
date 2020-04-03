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

# mid-rise uniform quantization
def quantize_weights_midRise(x, k):
    #mini = tf.reduce_min(x)
    #maxi = tf.reduce_max(x)
    #x = (x - mini)/(maxi - mini)
    G = tf.get_default_graph()
    n = float(2**k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

def vgg16net_noisy(input_node, netparams, err_mean, err_stddev, train_vars):
	weights_noisy, biases_noisy, err_w, err_b = helper.add_noise(netparams['weights'], netparams['biases'], err_mean, err_stddev, train_vars)
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	err_lyr = {}
	layers_err  = {}
	data_spec = helper.get_data_spec('vgg16net')
	err_lyr['input'] = tf.get_variable(name='input_lyr_err', shape=(1, data_spec.crop_size, data_spec.crop_size, data_spec.channels), initializer=tf.random_normal_initializer(mean=err_mean[0], stddev=err_stddev[0]), trainable=train_vars[0])
	input_node_noisy = tf.add(input_node, err_lyr['input'])
	conv1_1 = conv(input_node_noisy, weights_noisy['conv1_1'], biases_noisy['conv1_1'], 1, 1)
	err_lyr['conv1_1'] = tf.get_variable(name='conv1_1_lyr_err', shape=conv1_1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1_1'] = tf.add(conv1_1, err_lyr['conv1_1'])
	conv1_2 = conv(layers_err['conv1_1'], weights_noisy['conv1_2'], biases_noisy['conv1_2'], 1, 1)
	err_lyr['conv1_2'] = tf.get_variable(name='conv1_2_lyr_err', shape=conv1_2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1_2'] = tf.add(conv1_2, err_lyr['conv1_2'])
	pool1 = max_pool(layers_err['conv1_2'], 2, 2, 2, 2)
	conv2_1 = conv(pool1, weights_noisy['conv2_1'], biases_noisy['conv2_1'], 1, 1)
	err_lyr['conv2_1'] = tf.get_variable(name='conv2_1_lyr_err', shape=conv2_1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv2_1'] = tf.add(conv2_1, err_lyr['conv2_1'])
	conv2_2 = conv(layers_err['conv2_1'], weights_noisy['conv2_2'], biases_noisy['conv2_2'], 1, 1)
	err_lyr['conv2_2'] = tf.get_variable(name='conv2_2_lyr_err', shape=conv2_2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv2_2'] = tf.add(conv2_2, err_lyr['conv2_2'])
	pool2 = max_pool(layers_err['conv2_2'], 2, 2, 2, 2)
	conv3_1 = conv(pool2, weights_noisy['conv3_1'], biases_noisy['conv3_1'], 1, 1)
	err_lyr['conv3_1'] = tf.get_variable(name='conv3_1_lyr_err', shape=conv3_1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv3_1'] = tf.add(conv3_1, err_lyr['conv3_1'])
	conv3_2 = conv(layers_err['conv3_1'], weights_noisy['conv3_2'], biases_noisy['conv3_2'], 1, 1)
	err_lyr['conv3_2'] = tf.get_variable(name='conv3_2_lyr_err', shape=conv3_2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv3_2'] = tf.add(conv3_2, err_lyr['conv3_2'])
	conv3_3 = conv(layers_err['conv3_2'], weights_noisy['conv3_3'], biases_noisy['conv3_3'], 1, 1)
	err_lyr['conv3_3'] = tf.get_variable(name='conv3_3_lyr_err', shape=conv3_3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv3_3'] = tf.add(conv3_3, err_lyr['conv3_3'])
	pool3 = max_pool(layers_err['conv3_3'], 2, 2, 2, 2)
	conv4_1 = conv(pool3, weights_noisy['conv4_1'], biases_noisy['conv4_1'], 1, 1)
	err_lyr['conv4_1'] = tf.get_variable(name='conv4_1_lyr_err', shape=conv4_1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv4_1'] = tf.add(conv4_1, err_lyr['conv4_1'])
	conv4_2 = conv(layers_err['conv4_1'], weights_noisy['conv4_2'], biases_noisy['conv4_2'], 1, 1)
	err_lyr['conv4_2'] = tf.get_variable(name='conv4_2_lyr_err', shape=conv4_2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv4_2'] = tf.add(conv4_2, err_lyr['conv4_2'])
	conv4_3 = conv(layers_err['conv4_2'], weights_noisy['conv4_3'], biases_noisy['conv4_3'], 1, 1)
	err_lyr['conv4_3'] = tf.get_variable(name='conv4_3_lyr_err', shape=conv4_3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv4_3'] = tf.add(conv4_3, err_lyr['conv4_3'])
	pool4 = max_pool(layers_err['conv4_3'], 2, 2, 2, 2)
	conv5_1 = conv(pool4, weights_noisy['conv5_1'], biases_noisy['conv5_1'], 1, 1)
	err_lyr['conv5_1'] = tf.get_variable(name='conv5_1_lyr_err', shape=conv5_1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv5_1'] = tf.add(conv5_1, err_lyr['conv5_1'])
	conv5_2 = conv(layers_err['conv5_1'], weights_noisy['conv5_2'], biases_noisy['conv5_2'], 1, 1)
	err_lyr['conv5_2'] = tf.get_variable(name='conv5_2_lyr_err', shape=conv5_2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv5_2'] = tf.add(conv5_2, err_lyr['conv5_2'])
	conv5_3 = conv(layers_err['conv5_2'], weights_noisy['conv5_3'], biases_noisy['conv5_3'], 1, 1)
	err_lyr['conv5_3'] = tf.get_variable(name='conv5_3_lyr_err', shape=conv5_3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv5_3'] = tf.add(conv5_3, err_lyr['conv5_3'])
	pool5 = max_pool(layers_err['conv5_3'], 2, 2, 2, 2)
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
	
def vgg16net(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	data_spec = get_data_spec('vgg16net')
	conv1_1 = conv(input_node, weights['conv1_1'], biases['conv1_1'], 1, 1)
	conv1_2 = conv(conv1_1, weights['conv1_2'], biases['conv1_2'], 1, 1)
	pool1 = max_pool(conv1_2, 2, 2, 2, 2)
	conv2_1 = conv(pool1, weights['conv2_1'], biases['conv2_1'], 1, 1)
	conv2_2 = conv(conv2_1, weights['conv2_2'], biases['conv2_2'], 1, 1)
	pool2 = max_pool(conv2_2, 2, 2, 2, 2)
	conv3_1 = conv(pool2, weights['conv3_1'], biases['conv3_1'], 1, 1)
	conv3_2 = conv(conv3_1, weights['conv3_2'], biases['conv3_2'], 1, 1)
	conv3_3 = conv(conv3_2, weights['conv3_3'], biases['conv3_3'], 1, 1)
	pool3 = max_pool(conv3_3, 2, 2, 2, 2)
	conv4_1 = conv(pool3, weights['conv4_1'], biases['conv4_1'], 1, 1)
	conv4_2 = conv(conv4_1, weights['conv4_2'], biases['conv4_2'], 1, 1)
	conv4_3 = conv(conv4_2, weights['conv4_3'], biases['conv4_3'], 1, 1)
	pool4 = max_pool(conv4_3, 2, 2, 2, 2)
	conv5_1 = conv(pool4, weights['conv5_1'], biases['conv5_1'], 1, 1)
	conv5_2 = conv(conv5_1, weights['conv5_2'], biases['conv5_2'], 1, 1)
	conv5_3 = conv(conv5_2, weights['conv5_3'], biases['conv5_3'], 1, 1)
	pool5 = max_pool(conv5_3, 2, 2, 2, 2)
	fc6 = fc(pool5, weights['fc6'], biases['fc6'])
	fc7 = fc(fc6, weights['fc7'], biases['fc7'])
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8

def vgg16net_q(input_node, netparams, qbits):
	weights, biases = netparams['weights'], netparams['biases']
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	data_spec = get_data_spec('vgg16net')
	conv1_1 = conv(input_node, weights['conv1_1'], biases['conv1_1'], 1, 1)

	k = qbits[1]
	weights_conv1_2_q = quantize_weights(weights['conv1_2'], k)
	conv1_2 = conv(conv1_1, weights_conv1_2_q, biases['conv1_2'], 1, 1)
	pool1 = max_pool(conv1_2, 2, 2, 2, 2)
	
	k = qbits[2]
	weights_conv2_1_q = quantize_weights(weights['conv2_1'], k)
	conv2_1 = conv(pool1, weights_conv2_1_q, biases['conv2_1'], 1, 1)
	
	k = qbits[3]
	weights_conv2_2_q = quantize_weights(weights['conv2_2'], k)
	conv2_2 = conv(conv2_1, weights_conv2_2_q, biases['conv2_2'], 1, 1)
	pool2 = max_pool(conv2_2, 2, 2, 2, 2)
	
	k = qbits[4]
	weights_conv3_1_q = quantize_weights(weights['conv3_1'], k)
	conv3_1 = conv(pool2, weights_conv3_1_q, biases['conv3_1'], 1, 1)
	
	k = qbits[5]
	weights_conv3_2_q = quantize_weights(weights['conv3_2'], k)
	conv3_2 = conv(conv3_1, weights_conv3_2_q, biases['conv3_2'], 1, 1)
	
	k = qbits[6]
	weights_conv3_3_q = quantize_weights(weights['conv3_3'], k)
	conv3_3 = conv(conv3_2, weights_conv3_3_q, biases['conv3_3'], 1, 1)
	pool3 = max_pool(conv3_3, 2, 2, 2, 2)
	
	k = qbits[7]
	weights_conv4_1_q = quantize_weights(weights['conv4_1'], k)
	conv4_1 = conv(pool3, weights_conv4_1_q, biases['conv4_1'], 1, 1)
	
	k = qbits[8]
	weights_conv4_2_q = quantize_weights(weights['conv4_2'], k)
	conv4_2 = conv(conv4_1, weights_conv4_2_q, biases['conv4_2'], 1, 1)
	
	k = qbits[9]
	weights_conv4_3_q = quantize_weights(weights['conv4_3'], k)
	conv4_3 = conv(conv4_2, weights_conv4_3_q, biases['conv4_3'], 1, 1)
	pool4 = max_pool(conv4_3, 2, 2, 2, 2)
	
	k = qbits[10]
	weights_conv5_1_q = quantize_weights(weights['conv5_1'], k)
	conv5_1 = conv(pool4, weights_conv5_1_q, biases['conv5_1'], 1, 1)
	
	k = qbits[11]
	weights_conv5_2_q = quantize_weights(weights['conv5_2'], k)
	conv5_2 = conv(conv5_1, weights_conv5_2_q, biases['conv5_2'], 1, 1)
	
	k = qbits[12]
	weights_conv5_3_q = quantize_weights(weights['conv5_3'], k)
	conv5_3 = conv(conv5_2, weights_conv5_3_q, biases['conv5_3'], 1, 1)
	pool5 = max_pool(conv5_3, 2, 2, 2, 2)
	
	k = qbits[13]
	weights_fc6_q = quantize_weights(weights['fc6'], k)
	fc6 = fc(pool5, weights_fc6_q, biases['fc6'])
	
	k = qbits[14]
	weights_fc7_q = quantize_weights(weights['fc7'], k)
	fc7 = fc(fc6, weights_fc7_q, biases['fc7'])
	
	fc8 = fc(fc7, weights['fc8'], biases['fc8'], relu=False)
	return fc8
