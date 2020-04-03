import numpy as np
import tensorflow as tf
import pickle
import dataset
import helper
from helper import *

def nin_noisy(input_node, netparams, err_mean, err_stddev, train_vars):
	weights_noisy, biases_noisy, err_w, err_b = helper.add_noise(netparams['weights'], netparams['biases'], err_mean, err_stddev, train_vars)
	mean, variance, scale, offset = netparams['mean'], netparams['variance'], netparams['scale'], netparams['offset']
	err_lyr = {}
	layers_err  = {}
	data_spec = helper.get_data_spec('nin')
	err_lyr['input'] = tf.get_variable(name='input_lyr_err', shape=(1, data_spec.crop_size, data_spec.crop_size, data_spec.channels), initializer=tf.random_normal_initializer(mean=err_mean[0], stddev=err_stddev[0]), trainable=train_vars[0])
	input_node_noisy = tf.add(input_node, err_lyr['input'])
	conv1 = conv(input_node_noisy, weights_noisy['conv1'], biases_noisy['conv1'], 4, 4, padding='VALID')
	err_lyr['conv1'] = tf.get_variable(name='conv1_lyr_err', shape=conv1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv1'] = tf.add(conv1, err_lyr['conv1'])
	cccp1 = conv(layers_err['conv1'], weights_noisy['cccp1'], biases_noisy['cccp1'], 1, 1)
	err_lyr['cccp1'] = tf.get_variable(name='cccp1_lyr_err', shape=cccp1.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp1'] = tf.add(cccp1, err_lyr['cccp1'])
	cccp2 = conv(layers_err['cccp1'], weights_noisy['cccp2'], biases_noisy['cccp2'], 1, 1)
	err_lyr['cccp2'] = tf.get_variable(name='cccp2_lyr_err', shape=cccp2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp2'] = tf.add(cccp2, err_lyr['cccp2'])
	pool1 = max_pool(layers_err['cccp2'], 3, 3, 2, 2)
	conv2 = conv(pool1, weights_noisy['conv2'], biases_noisy['conv2'], 1, 1)
	err_lyr['conv2'] = tf.get_variable(name='conv2_lyr_err', shape=conv2.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv2'] = tf.add(conv2, err_lyr['conv2'])
	cccp3 = conv(layers_err['conv2'], weights_noisy['cccp3'], biases_noisy['cccp3'], 1, 1)
	err_lyr['cccp3'] = tf.get_variable(name='cccp3_lyr_err', shape=cccp3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp3'] = tf.add(cccp3, err_lyr['cccp3'])
	cccp4 = conv(layers_err['cccp3'], weights_noisy['cccp4'], biases_noisy['cccp4'], 1, 1)
	err_lyr['cccp4'] = tf.get_variable(name='cccp4_lyr_err', shape=cccp4.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp4'] = tf.add(cccp4, err_lyr['cccp4'])
	pool2 = max_pool(layers_err['cccp4'], 3, 3, 2, 2, padding='VALID')
	conv3 = conv(pool2, weights_noisy['conv3'], biases_noisy['conv3'], 1, 1)
	err_lyr['conv3'] = tf.get_variable(name='conv3_lyr_err', shape=conv3.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv3'] = tf.add(conv3, err_lyr['conv3'])
	cccp5 = conv(layers_err['conv3'], weights_noisy['cccp5'], biases_noisy['cccp5'], 1, 1)
	err_lyr['cccp5'] = tf.get_variable(name='cccp5_lyr_err', shape=cccp5.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp5'] = tf.add(cccp5, err_lyr['cccp5'])
	cccp6 = conv(layers_err['cccp5'], weights_noisy['cccp6'], biases_noisy['cccp6'], 1, 1)
	err_lyr['cccp6'] = tf.get_variable(name='cccp6_lyr_err', shape=cccp6.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp6'] = tf.add(cccp6, err_lyr['cccp6'])
	pool3 = max_pool(layers_err['cccp6'], 3, 3, 2, 2, padding='VALID')
	conv4_1024 = conv(pool3, weights_noisy['conv4_1024'], biases_noisy['conv4_1024'], 1, 1)
	err_lyr['conv4_1024'] = tf.get_variable(name='conv4_1024_lyr_err', shape=conv4_1024.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['conv4_1024'] = tf.add(conv4_1024, err_lyr['conv4_1024'])
	cccp7_1024 = conv(layers_err['conv4_1024'], weights_noisy['cccp7_1024'], biases_noisy['cccp7_1024'], 1, 1)
	err_lyr['cccp7_1024'] = tf.get_variable(name='cccp7_1024_lyr_err', shape=cccp7_1024.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp7_1024'] = tf.add(cccp7_1024, err_lyr['cccp7_1024'])
	cccp8_1024 = conv(layers_err['cccp7_1024'], weights_noisy['cccp8_1024'], biases_noisy['cccp8_1024'], 1, 1)
	err_lyr['cccp8_1024'] = tf.get_variable(name='cccp8_1024_lyr_err', shape=cccp8_1024.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])
	layers_err['cccp8_1024'] = tf.add(cccp8_1024, err_lyr['cccp8_1024'])
	pool4 = avg_pool(layers_err['cccp8_1024'], 6, 6, 1, 1, padding='VALID')
	return pool4, err_w, err_b, err_lyr
	