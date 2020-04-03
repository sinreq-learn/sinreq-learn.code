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


def svhn_net(input_node, netparams):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('svhn_net')
	conv1 = conv(input_node, weights['hidden1'], biases['hidden1'], 1, 1, padding='SAME', relu=True)
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 2, 2, 2, 2, padding='SAME')

	conv2 = conv(pool1, weights['hidden2'], biases['hidden2'], 1, 1, relu=True)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 2, 2, 1, 1, padding='SAME')

	conv3 = conv(pool2, weights['hidden3'], biases['hidden3'], 1, 1, relu=True)
	norm3 = lrn(conv3, 2, 1.99999994948e-05, 0.75)
	pool3 = max_pool(norm3, 2, 2, 2, 2, padding='SAME')

	conv4 = conv(pool3, weights['hidden4'], biases['hidden4'], 1, 1, relu=True)
	norm4 = lrn(conv4, 2, 1.99999994948e-05, 0.75)
	pool4 = max_pool(norm4, 2, 2, 1, 1, padding='SAME')

	conv5 = conv(pool4, weights['hidden5'], biases['hidden5'], 1, 1, relu=True)
	norm5 = lrn(conv5, 2, 1.99999994948e-05, 0.75)
	pool5 = max_pool(norm5, 2, 2, 2, 2, padding='SAME')

	conv6 = conv(pool5, weights['hidden6'], biases['hidden6'], 1, 1, relu=True)
	norm6 = lrn(conv6, 2, 1.99999994948e-05, 0.75)
	pool6 = max_pool(norm6, 2, 2, 1, 1, padding='SAME')

	conv7 = conv(pool6, weights['hidden7'], biases['hidden7'], 1, 1, relu=True)
	norm7 = lrn(conv7, 2, 1.99999994948e-05, 0.75)
	pool7 = max_pool(norm7, 2, 2, 2, 2, padding='SAME')

	conv8 = conv(pool7, weights['hidden8'], biases['hidden8'], 1, 1, relu=True)
	norm8 = lrn(conv8, 2, 1.99999994948e-05, 0.75)
	pool8 = max_pool(norm8, 2, 2, 1, 1, padding='SAME')

	flatten = tf.reshape(pool8, [-1, 4 * 4 * 192])

	hidden9 = fc(flatten, weights['hidden9'], biases['hidden9'])
	
	hidden10 = fc(hidden9, weights['hidden10'], biases['hidden10'])
	
	length = fc(hidden10, weights['digit_length'], biases['digit_length'])

	digit1 = fc(hidden10, weights['digit1'], biases['digit1'])

	digit2 = fc(hidden10, weights['digit2'], biases['digit2'])

	digit3 = fc(hidden10, weights['digit3'], biases['digit3'])

	digit4 = fc(hidden10, weights['digit4'], biases['digit4'])

	digit5 = fc(hidden10, weights['digit5'], biases['digit5'])

	length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
	#print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	#print(conv1)
	#print(flatten)
	return length_logits, digits_logits

def svhn_net_q(input_node, netparams, qbits_dict):
	weights, biases = netparams['weights'], netparams['biases']
	data_spec = get_data_spec('svhn_net')
	conv1 = conv(input_node, weights['hidden1'], biases['hidden1'], 1, 1, padding='SAME', relu=True)
	norm1 = lrn(conv1, 2, 1.99999994948e-05, 0.75)
	pool1 = max_pool(norm1, 2, 2, 2, 2, padding='SAME')


	k = qbits_dict['hidden2']
	weights_hidden2 = quantize_weights(weights['hidden2'], k)
	conv2 = conv(pool1, weights_hidden2, biases['hidden2'], 1, 1, relu=True)
	norm2 = lrn(conv2, 2, 1.99999994948e-05, 0.75)
	pool2 = max_pool(norm2, 2, 2, 1, 1, padding='SAME')

	k = qbits_dict['hidden3']
	weights_hidden3 = quantize_weights(weights['hidden3'], k)
	conv3 = conv(pool2, weights_hidden3, biases['hidden3'], 1, 1, relu=True)
	norm3 = lrn(conv3, 2, 1.99999994948e-05, 0.75)
	pool3 = max_pool(norm3, 2, 2, 2, 2, padding='SAME')


	k = qbits_dict['hidden4']
	weights_hidden4 = quantize_weights(weights['hidden4'], k)
	conv4 = conv(pool3, weights_hidden4, biases['hidden4'], 1, 1, relu=True)
	norm4 = lrn(conv4, 2, 1.99999994948e-05, 0.75)
	pool4 = max_pool(norm4, 2, 2, 1, 1, padding='SAME')


	k = qbits_dict['hidden5']
	weights_hidden5 = quantize_weights(weights['hidden5'], k)
	conv5 = conv(pool4, weights_hidden5, biases['hidden5'], 1, 1, relu=True)
	norm5 = lrn(conv5, 2, 1.99999994948e-05, 0.75)
	pool5 = max_pool(norm5, 2, 2, 2, 2, padding='SAME')


	k = qbits_dict['hidden6']
	weights_hidden6 = quantize_weights(weights['hidden6'], k)
	conv6 = conv(pool5, weights_hidden6, biases['hidden6'], 1, 1, relu=True)
	norm6 = lrn(conv6, 2, 1.99999994948e-05, 0.75)
	pool6 = max_pool(norm6, 2, 2, 1, 1, padding='SAME')

	k = qbits_dict['hidden7']
	weights_hidden7 = quantize_weights(weights['hidden7'], k)
	conv7 = conv(pool6, weights_hidden7, biases['hidden7'], 1, 1, relu=True)
	norm7 = lrn(conv7, 2, 1.99999994948e-05, 0.75)
	pool7 = max_pool(norm7, 2, 2, 2, 2, padding='SAME')

	k = qbits_dict['hidden8']
	weights_hidden8 = quantize_weights(weights['hidden8'], k)
	conv8 = conv(pool7, weights_hidden8, biases['hidden8'], 1, 1, relu=True)
	norm8 = lrn(conv8, 2, 1.99999994948e-05, 0.75)
	pool8 = max_pool(norm8, 2, 2, 1, 1, padding='SAME')

	flatten = tf.reshape(pool8, [-1, 4 * 4 * 192])

	k = qbits_dict['hidden9']
	weights_hidden9 = quantize_weights(weights['hidden9'], k)
	hidden9 = fc(flatten, weights_hidden9, biases['hidden9'])
	
	hidden10 = fc(hidden9, weights['hidden10'], biases['hidden10'])
	
	length = fc(hidden10, weights['digit_length'], biases['digit_length'])

	digit1 = fc(hidden10, weights['digit1'], biases['digit1'])

	digit2 = fc(hidden10, weights['digit2'], biases['digit2'])

	digit3 = fc(hidden10, weights['digit3'], biases['digit3'])

	digit4 = fc(hidden10, weights['digit4'], biases['digit4'])

	digit5 = fc(hidden10, weights['digit5'], biases['digit5'])

	length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
	#print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	#print(conv1)
	#print(flatten)
	return length_logits, digits_logits
