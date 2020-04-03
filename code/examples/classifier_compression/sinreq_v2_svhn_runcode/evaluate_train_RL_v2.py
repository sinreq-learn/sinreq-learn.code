#!/usr/bin/python

# developer: Ahmed Taha Elthakeb
# email: (a1yousse@eng.ucsd.edu)

"""
[21 OCT 2018]
- testcase: alexnet 
- changing the reward function to be func(val_acc + finetune_acc) on 10k images 
"""

from __future__ import division
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
#import dataset

#from . import dataset, networks, load
import sys
sys.path.append("..")
import dataset, networks, load, quantize

#import networks
#from networks import *
from networks import helper
from networks import alexnet, resnet18
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
#import load
import json
from quantize import quantize_network
import six
import csv
import math

"""
from __future__ import division
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
#import dataset

#from . import dataset, networks, load
import sys
sys.path.append("..")
import dataset, networks, load, quantize

#import networks
#from networks import *
from networks import helper
from networks import alexnet, resnet18
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
#import load
import json
from quantize import quantize_network
import six
"""
file_idx = 1

#NETWORKS = ['alexnet', 'googlenet', 'nin', 'resnet18', 'resnet50', 'squeezenet', 'vgg16net', 'lenet']
NETWORKS = ['lenet']
IMAGE_PATH_TRAIN = '/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_40K/'
IMAGE_PATH_TEST = '/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_10K/'
IMAGE_LABLE = '/home/ahmed/projects/NN_quant/rlbitwidth.code/val.txt'
CKPT_PATH = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/'
NET_ACC = {'alexnet': 79.918, 'googlenet': 89.002, 'nin': 81.218, 'resnet18': 85.016, 
	'resnet50': 91.984, 'squeezenet': 80.346, 'vgg16net': 89.816, 'lenet': 99.06}

HOME = '/home/ahmed/projects/rl_quantization/rl_quantization.code'

def eval_imagenet(net_name, param_path, param_q_path, qbits, layer_index, layer_name, file_idx, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=200., n_epoch=1):
	"""all layers are trainable in the conventional retraining procedure"""
	if '.ckpt' in param_path:
		netparams = load.load_netparams_tf(param_path, trainable=True)
	else:
		netparams = load.load_netparams_tf_q(param_path, trainable=True)

	data_spec = helper.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.int32)

	if net_name == 'alexnet_noisy':
		logits_, err_w, err_b, err_lyr = networks.alexnet_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'alexnet':
		if trainable:
			logits_ = alexnet.alexnet_q(input_node, netparams, qbits)
		else:
			logits_ = alexnet.alexnet(input_node, netparams)
	elif net_name == 'alexnet_shift':
		logits_ = networks.alexnet_shift(input_node, netparams)
	elif net_name == 'googlenet':
		logits_, err_w, err_b, err_lyr = networks.googlenet_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'nin':
		logits_, err_w, err_b, err_lyr = networks.nin_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'resnet18':
		logits_ = resnet18.resnet18(input_node, netparams)
		#logits_, err_w, err_b, err_lyr = networks.resnet18_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'resnet18_shift':
		logits_ = networks.resnet18_shift(input_node, netparams, shift_back)
	elif net_name == 'resnet50':
		logits_, err_w, err_b, err_lyr = networks.resnet50_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'squeezenet':
		logits_, err_w, err_b, err_lyr = networks.squeezenet_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'vgg16net':
		logits_, err_w, err_b, err_lyr = networks.vgg16net_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	
	#square = [tf.nn.l2_loss(err_w[layer]) for layer in err_w]
	#square_sum = tf.reduce_sum(square)
	#loss_op = tf.reduce_mean(tf.nn.oftmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor / (1. + square_sum)
	
	# ======== calculating the quantization error of a certain layer ==========
	if trainable:
		""" read the quantized weights (quantized version of the most recent retrained) """
		w_q_pickle = param_q_path
		with open(w_q_pickle, 'rb') as f:
			params_quantized = pickle.load(f)
		
		layer = layer_name
		params_quantized_layer = tf.get_variable(name='params_quantized_layer', initializer=tf.constant(params_quantized[0][layer]), trainable=False)
		
		q_diff = tf.subtract(params_quantized_layer, netparams['weights'][layer])
		q_diff_cost = tf.nn.l2_loss(q_diff)
		#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor*q_diff_cost

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) 

	probs  = helper.softmax(logits_)
	top_k_op = tf.nn.in_top_k(probs, label_node, 5)
	#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.1)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10-8)
	if trainable:
	    train_op = optimizer.minimize(loss_op)
	correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(label_node, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if trainable:
			count = 0
			correct = 0
			cur_accuracy = 0
			for i in range(0, n_epoch):
				#if cur_accuracy >= NET_ACC[net_name]:
						#break
				#image_producer = dataset.ImageNetProducer(val_path='/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_10K/val_10.txt', data_path='/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_10K', data_spec=data_spec)
				#path_train = '/home/ahmed/projects/NN_quant/imageNet_training'
				path_train = '/home/ahmed/ILSVRC2012_img_train'
				
				#image_producer = dataset.ImageNetProducer(val_path=path_train + '/train_shuf_'+str(file_idx)+'.txt', data_path=path_train, data_spec=data_spec)
				image_producer = dataset.ImageNetProducer(val_path=path_train + '/reward_20k.txt', data_path=path_train, data_spec=data_spec)
				total = len(image_producer) * n_epoch
				coordinator = tf.train.Coordinator()
				threads = image_producer.start(session=sess, coordinator=coordinator)
				for (labels, images) in image_producer.batches(sess):
					one_hot_labels = np.zeros((len(labels), 1000))
					for k in range(len(labels)):
						one_hot_labels[k][labels[k]] = 1
					sess.run(train_op, feed_dict={input_node: images, label_node: one_hot_labels})
					
					# AHMED: debug
					#netparams_tmp = sess.run(netparams)
					#print('train = ', np.amax(netparams_tmp['weights']['conv2']))
					#print('len set = ', len(set(np.array(netparams['weights']['conv2']))))
					# ------------
					
					#correct += np.sum(sess.run(top_k_op, feed_dict={input_node: images, label_node: labels}))
					# AHMED: modify 
					#top, logits_tmp, loss_op_tmp = sess.run([top_k_op, logits_q, loss_op], feed_dict={input_node: images, label_node: labels})
					#top, act_q_tmp, weights_fp_tmp, weights_q_tmp = sess.run([top_k_op, act_, weights_fp, weights_q], feed_dict={input_node: images, label_node: labels})
					top = sess.run([top_k_op], feed_dict={input_node: images, label_node: labels})
					correct += np.sum(top)
					#print(np.amax(weights_q_tmp))
					#print(len(set(weights_q_tmp.ravel())))
					# --------
					count += len(labels)
					cur_accuracy = float(correct) * 100 / count
					
					file_label = 'retrain_reward'
					for i in range(len(qbits)):
						file_label=file_label+'_'+str(qbits[i])
					write_to_csv([count, total, cur_accuracy],file_label)
					
					print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
				coordinator.request_stop()
				coordinator.join(threads, stop_grace_period_secs=2)
			#return sess.run(err_w), cur_accuracy
			# "sess.run" returns the netparams as normal value (converts it from tf to normal python variable)
			return cur_accuracy, sess.run(netparams)
		else:
			count = 0
			correct = 0
			cur_accuracy = 0
			#path_val = './nn_quant_and_run_code_train/ILSVRC2012_img_val'
			path_val = '/home/ahmed/ILSVRC2012_img_val'
			image_producer = dataset.ImageNetProducer(val_path=path_val + '/val_reward_3k.txt', data_path=path_val, data_spec=data_spec)
			#image_producer = dataset.ImageNetProducer(val_path='/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_40K/val_40.txt', data_path='/home/ahmed/projects/NN_quant/ILSVRC2012_img_val_40K', data_spec=data_spec)
			total = len(image_producer)
			coordinator = tf.train.Coordinator()
			threads = image_producer.start(session=sess, coordinator=coordinator)
			for (labels, images) in image_producer.batches(sess):
				one_hot_labels = np.zeros((len(labels), 1000))
				for k in range(len(labels)):
					one_hot_labels[k][labels[k]] = 1
				#correct += np.sum(sess.run(top_k_op, feed_dict={input_node: images, label_node: labels}))
				top = sess.run([top_k_op], feed_dict={input_node: images, label_node: labels})
				correct += np.sum(top)
				count += len(labels)
				cur_accuracy = float(correct) * 100 / count
				print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
			coordinator.request_stop()
			coordinator.join(threads, stop_grace_period_secs=2)
			return cur_accuracy, 0

def eval_imagenet_q(net_name, param_pickle_path):
	netparams = load.load_netparams_tf_q(param_pickle_path)
	data_spec = helper.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.int32)
	if net_name == 'alexnet':
		logits_ = alexnet.alexnet(input_node, netparams)
	elif net_name == 'googlenet':
		logits_ = networks.googlenet(input_node, netparams)
	elif net_name == 'nin':
		logits_ = networks.nin(input_node, netparams)
	elif net_name == 'resnet18':
		logits_ = networks.resnet18(input_node, netparams)
	elif net_name == 'resnet50':
		logits_ = networks.resnet50(input_node, netparams)
	elif net_name == 'squeezenet':
		logits_ = networks.squeezenet(input_node, netparams)
	elif net_name == 'vgg16net':
		logits_ = networks.vgg16net_noisy(input_node, netparams)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) # 
	probs  = softmax(logits_)
	top_k_op = tf.nn.in_top_k(probs, label_node, 5)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.1)
	correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(label_node, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	count = 0
	correct = 0
	cur_accuracy = 0
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		image_producer = dataset.ImageNetProducer(val_path=IMAGE_LABLE, data_path=IMAGE_PATH, data_spec=data_spec)
		total = len(image_producer)
		coordinator = tf.train.Coordinator()
		threads = image_producer.start(session=sess, coordinator=coordinator)
		for (labels, images) in image_producer.batches(sess):
			correct += np.sum(sess.run(top_k_op, feed_dict={input_node: images, label_node: labels}))
			count += len(labels)
			cur_accuracy = float(correct) * 100 / count
			print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
		print(cur_accuracy)
		coordinator.request_stop()
		coordinator.join(threads, stop_grace_period_secs=2)
	return cur_accuracy
	
		
def eval_lenet(net_name, ckpt_path, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=200., n_epoch=1):
	netparams = load.load_netparams_tf(ckpt_path, trainable=False)
	data_spec = networks.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size * data_spec.crop_size * data_spec.channels))
	input_node_2d = tf.reshape(input_node, shape=(-1, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.float32, [None, 10])
	logits_, err_w, err_b, err_lyr = lenet_noisy(input_node_2d, netparams, err_mean, err_stddev, train_vars)
	square = [tf.nn.l2_loss(err_w[layer]) for layer in err_w]
	square_sum = tf.reduce_sum(square)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor / (1. + square_sum)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	if trainable:
		train_op = optimizer.minimize(loss_op)
	probs  = softmax(logits_)
	correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(label_node, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	mnist = mnist_input.read_data_sets("/tmp/data/", one_hot=True)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		cur_accuracy = 0
		for i in range(0, n_epoch):
			#if cur_accuracy >= NET_ACC[net_name]:
					#break
			if trainable:
				for step in range(0, mnist.train.num_examples/data_spec.batch_size):
					batch_x, batch_y = mnist.train.next_batch(data_spec.batch_size)
					sess.run(train_op, feed_dict={input_node: batch_x, label_node: batch_y})
					#loss, acc = sess.run([loss_op_1, accuracy], feed_dict={input_node: batch_x, label_node: batch_y})
				print('Training finished\n')
			cur_accuracy = 100 * (sess.run(accuracy, feed_dict={input_node: mnist.test.images[:], label_node: mnist.test.labels[:]}))
			print('test accuracy:\t' + (str)(cur_accuracy))
		return sess.run(err_w), cur_accuracy
				
def run_network(net_name, cost_factor, n_epoch):
	ckpt_path = CKPT_PATH + net_name + '/' + net_name + '.ckpt'
	err_mean =   [0.0, 0.0, 0.0, 0.0] #order: input, weights, biases, layers
	err_stddev = [0.0, 0.0, 0.0, 0.0]
	train_vars = [False, True, False, False]
	istrain = True
	if net_name == 'lenet':
		return eval_lenet(net_name, ckpt_path, trainable=istrain, err_mean=err_mean, err_stddev=err_stddev, train_vars=train_vars, cost_factor=cost_factor, n_epoch=n_epoch)
	else:
		return eval_imagenet(net_name, ckpt_path, trainable=istrain, err_mean=err_mean, err_stddev=err_stddev, train_vars=train_vars, cost_factor=cost_factor, n_epoch=n_epoch)

def gen_max_noise_dist():
	max_epoch = 5
	for net_name in NETWORKS:
		directory = '/home/ahmed/projects/NN_quant/results/networks/' + net_name
		if not os.path.exists(directory):
			os.makedirs(directory)
		current_factor = 10
		largest_correct = 0
		smallest_wrong = 0
		for i in range(0, 10):
			tf.reset_default_graph()
			err_w, accuracy = run_network(net_name, current_factor, max_epoch)
			if accuracy >= NET_ACC[net_name]:
				save_path = directory + '/' + (str)(current_factor) + '_' + (str)(accuracy)
				with open(save_path, 'w') as f:
					pickle.dump(err_w, f)
				largest_correct = current_factor
				if smallest_wrong == 0:
					current_factor = current_factor * 2
				else:
					current_factor = (current_factor + smallest_wrong) / 2.
			else:
				smallest_wrong = current_factor
				current_factor = (current_factor + largest_correct) / 2.

def gen_noise_dist(net_name, cost_factor, count, n_epoch):
	directory = '/home/ahmed/projects/NN_quant/results/deltas/' + net_name
	if not os.path.exists(directory):
		os.makedirs(directory)  
	for i in range(0, count):
		tf.reset_default_graph()
		err_w, accuracy = run_network(net_name, cost_factor, n_epoch)
		#save_path = directory + '/' + (str)(cost_factor) + '_' + (str)(accuracy)
		save_path = directory + '/' + (str)(i) + '_' + (str)(accuracy)
		with open(save_path, 'w') as f:
			pickle.dump(err_w, f)

'''
path_net = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py'
layers = load.get_layers(path_net)
acc = {}
for i in range(0, len(layers)):
	path = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_10_' + layers[i] + '_7.pickle'
	tf.reset_default_graph()
	acc[layers[i]] = eval_imagenet_q('resnet18', path)
	print("\n\n")
	print(str(i) + "/" + str(len(layers)))
	print(acc)
print(acc)
'''

def eval_normalized_layers():
	acc = {}
	count = 1
	for dirpath, subdirs, fileList in os.walk('/home/ahmed/projects/NN_quant/results/normalized/resnet18/'):
		for filename in fileList:
			addr = (os.path.join(dirpath, filename))
			tf.reset_default_graph()
			print()
			print(count)
			print(filename)
			print()
			count = count + 1
			acc[filename] = eval_imagenet_q('resnet18', addr)
	print(acc)
	with open('out.txt', 'w') as outfile:
		outfile.write(json.dumps(acc))

#layers_sorted = load.get_layers('/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
shift_back = {}
#for layer in layers_sorted:
#	shift_back[layer] = 0


#print('==================================================================')
#print('TRAINING')
#print('==================================================================')

'''
# this is for phase I training - retrain a little bit on new dataset 40K - get Wo'
#param_path = '/home/ahmed/projects/NN_quant/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt' # = {Wo}
# this is for phase II training - retrain to minimize the quantization error - >> get W1

#path_save = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18'
#path_save_q = path_save + '_layers_shift_quant_10May.pickle'
#param_path = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/May12_resnet18_10_fc1000_5_bits.pickle'
param_path = Wo_resent18
save_path_params = path_save + '_layers_shift_quant_retrain_A_10May.pickle'

acc, netparams = eval_imagenet('resnet18', param_path, shift_back, trainable=True, err_mean=None, err_stddev=None, train_vars=None, cost_factor=800., n_epoch=1)
print(acc)
with open(save_path_params, 'w') as f:
	pickle.dump(netparams, f)
'''

def get_stats(network_name):
	# get weights 
	netparams = load.get_netparams('./nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.ckpt')
	weights = netparams['weights']
	# get layers
	layers_sorted = load.get_layers('./nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.py')
	tot_num_layers = len(layers_sorted)
	cols = ['layer_idx_norm', 'n', 'c', 'k', 'std']
	tmp_lst = []
	for i, layer in enumerate(layers_sorted, start=1):
		layer_shape = weights[layer].shape
		if len(layer_shape) == 2:
			k = 0
			n, c = layer_shape
		else:
			k, _, n, c = layer_shape
		weights_layer = weights[layer].ravel()
		idx_norm = i/tot_num_layers
		std = np.var(weights_layer)
		tmp_lst.append([idx_norm, n, c, k, std])

	df = pd.DataFrame(tmp_lst, columns=cols)
	return df  # to access --> df.loc[i, 'std']      

def retrain():
	print('nothing here yet!')

def quantize_and_run(qbits):

	input_file = './nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
	
	""" Quantization """
	nbits = 16
	path_save = './nn_quant_and_run_code/results/quantized/alexnet/'
	path_save_q = path_save + 'alexnet_layers_quant_'+ str(nbits) +'-bits_23Sep.pickle'
	#layers_sorted = load.get_layers('/backup/amir-tc/rl_quantization/rl_quantization.code/nn_quant_and_run_code/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py')
	layers_sorted = load.get_layers('./nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py')
	#bits_q = [nbits] * len(layers_sorted)
	bits_q = qbits
	path_params = input_file
	quantize_network(path_params, layers_sorted, path_save_q, bits_q)

	print('==================================================================')
	print('INFERENCE')
	print('==================================================================')
	
	""" Run Inference """
	#path_save_q = path_save + '_layers_shift_quant_10May.pickle'
	#param_path = save_path_params
	#param_path = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_layers_shift_quant_retrain_A_10May.pickle'
	param_path = path_save_q
	with tf.Graph().as_default():
		acc, netparams = eval_imagenet('alexnet', param_path, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)

	return acc

def quantize_and_train(network_name, layer_index, layer_name, qbits, init_params, file_idx):
	""" full precision """
	#input_file = './rlbitwidth.tfmodels/caffe2tf/tfmodels/' + network_name + '/' + network_name + '.ckpt'
	global HOME

	print('==================================================================')
	print('Quantization')
	print('==================================================================')
	
	""" Quantization """
	""" 1) we initialize based on the quantized input pattern (?) """
	path_save = HOME+'/nn_quant_and_run_code/results/quantized/' + network_name + '/'
	path_save_q = path_save + 'train_1_init_' + network_name + '_layers_quant_17Oct.pickle'
	layers_sorted = load.get_layers(HOME+'/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/' + network_name + '/' + network_name + '.py')

	""" always start with the most recent retrained model """
	path_params = init_params

	quantize_network(path_params, layers_sorted, path_save_q, qbits)

	print('==================================================================')
	print('TRAINING')
	print('==================================================================')
	
	""" Run retraining """
	""" use the full precision weights for initialization, or the most recent retrained """
	""" this is used to calculate the quantization difference regularizer """
	param_path = init_params
	param_q_path = path_save_q

	with tf.Graph().as_default():
		acc, netparams = eval_imagenet(network_name, param_path, param_q_path, qbits, layer_index, layer_name, file_idx, shift_back, trainable=True, err_mean=None, err_stddev=None, train_vars=None, cost_factor=200., n_epoch=1)
	print(acc)

	""" path for saving the retrained model """
	path_save = HOME+'/nn_quant_and_run_code/results/quantized/' + network_name + '/' + network_name
	path_save_params = path_save + '_train_1_layers_quant_retrained_17Oct_RL.pickle'
	
	# AHMED: debug
	#print('retrained = ', np.amax(netparams['weights']['conv2']))
	#print('len set = ', len(set(np.array(netparams['weights']['conv2']))))
	# ------------
	
	with open(path_save_params, 'wb') as f:
		pickle.dump(netparams, f)

	print('==================================================================')
	print('TRAINING DONE!')
	print('==================================================================')
	
	return acc

def quantize_and_run_any(network, qbits):

	print('network:', network)
	input_file = './nn_quant_and_run_code/rlbitwidth.tfmodels/caffe2tf/tfmodels/' + network +'/' + network +'.ckpt'
	

	print('==================================================================')
	print('Quantization')
	print('==================================================================')
	
	""" Quantization """
	nbits = 10
	path_save = './nn_quant_and_run_code/results/quantized/'+ network +'/'
	path_save_q = path_save + network +'_layers_quant_'+ str(nbits) +'-bits_date.pickle'
	#layers_sorted = load.get_layers('/backup/amir-tc/rl_quantization/rl_quantization.code/nn_quant_and_run_code/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py')
	layers_sorted = load.get_layers('./nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+ network +'/'+ network +'.py')
	#bits_q = [nbits] * len(layers_sorted)
	bits_q = qbits
	path_params = input_file
	quantize_network(path_params, layers_sorted, path_save_q, bits_q)

	print('==================================================================')
	print('INFERENCE')
	print('==================================================================')
	
	""" Run Inference """
	#path_save_q = path_save + '_layers_shift_quant_10May.pickle'
	#param_path = save_path_params
	#param_path = '/home/ahmed/projects/NN_quant/results/quantized/resnet18/resnet18_layers_shift_quant_retrain_A_10May.pickle'
	param_path = path_save_q
	with tf.Graph().as_default():
		acc, netparams = eval_imagenet(network, param_path, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)
	return acc

def run_inference(network, input_param_path, qbits):

	global HOME

	print('==================================================================')
	print('Quantization')
	print('==================================================================')
	
	""" Quantization """
	nbits = 10
	path_save = HOME+'/nn_quant_and_run_code/results/quantized/'+ network +'/'
	path_save_q = path_save + network +'train_1_test_retrained_quantized.pickle'
	#layers_sorted = load.get_layers('/backup/amir-tc/rl_quantization/rl_quantization.code/nn_quant_and_run_code/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.py')
	layers_sorted = load.get_layers(HOME+'/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+ network +'/'+ network +'.py')
	#bits_q = [nbits] * len(layers_sorted)
	bits_q = qbits
	path_params = input_param_path
	quantize_network(path_params, layers_sorted, path_save_q, bits_q)

	print('==================================================================')
	print('INFERENCE')
	print('==================================================================')
	
	#param_path = input_param_path
	param_path = path_save_q
	param_q_path = ''
	layer_index = 0
	layer_name = 0
	file_idx = 0
	shift_back = {}
	with tf.Graph().as_default():
		acc, netparams = eval_imagenet(network, param_path, param_q_path, qbits, layer_index, layer_name, file_idx, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)
	return acc

def write_to_csv(step_data, file_label):
    with open('./retrain_slops_files/'+file_label+'.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(step_data)

# def main():



def retrain(network_name, episode_num, layer_index, qbits):

	global HOME
	"""
		1- read initial model (or the one from previous iteration) --> PARAMS(1)
		2- From RL: 
			- read layer index 
			- read #bits
		3- quantize: starting from "FP" (@ start of each episode) ->- quantize --> PARAMS_q
		3''- quantize: starting from "recent_retrained" ->- quantize --> "PARAMS_q"
		4- calculate the quantization error of the input layer: ||(FP(layer) - PARAMS_q(layer))||^2
		4''- calculate the quantization error of the input layer: ||(recent_retrained(layer) - PARAMS_q(layer))||^2
		5- add this quantization error to the objective function 
		6- initialize with PARAMS(1), fix previous layers (except 1st and last) and run retraining ... --> PARAMS_retrained
		7- (caching!!)
		* assume independent retraining for independent episodes 
	"""

	""" 
	- init_params = is the parameter file for retraining initialization
	- if starting the episode, then init_params comes from the full precision ckpt,
			otherwise, it comes from the most recent retrained file 
	"""
	global file_idx

	path_save = HOME + '/nn_quant_and_run_code/results/quantized/' + network_name + '/' + network_name
	path_params_retrained = path_save + '_train_1_layers_quant_retrained_17Oct_RL.pickle'
	#if path_params_retrained.is_file():

	""" - init_params = is the parameter file for retraining initialization
		- if starting the episode, then init_params comes from the full precision ckpt,
			otherwise, it comes from the most recent retrained file """
	#if (episode_num==0) and (layer_index==1):
	if (layer_index==1):
		init_params = HOME + '/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.ckpt'
		""" randomly pick 100k images to retrain on """
		#file_idx = random.randint(1,13)
		file_idx = 1
	else:
		init_params = path_params_retrained

	""" accelerated fine-tuning """
	layers_sorted = load.get_layers(HOME+'/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.py')
	layer_name = layers_sorted[layer_index]


	init_params = HOME + '/nn_quant_and_run_code_train/rlbitwidth.tfmodels/caffe2tf/tfmodels/'+network_name+'/'+network_name+'.ckpt'
	""" validation accuracy before fine-tuning """
	print('[a] validation ACC before retraining')
	print('##########################################################')
	#path_save = HOME+'/nn_quant_and_run_code/results/quantized/' + network_name + '/' + network_name
	#path_params_retrained = path_save + '_train_1_layers_quant_retrained_17Oct_RL.pickle'
	acc_val = run_inference(network_name, init_params, qbits)
	print(acc_val)

	print('[a] retraining ACC')
	print('##########################################################')
	acc_retrain = quantize_and_train(network_name, layer_index, layer_name, qbits, init_params, file_idx)

	""" path for saving the retrained model """
	path_save = HOME+'/nn_quant_and_run_code/results/quantized/' + network_name + '/' + network_name
	path_save_params = path_save + '_train_1_layers_quant_retrained_17Oct_RL.pickle'
	acc_retrain_val = run_inference(network_name, path_save_params, qbits)
	return acc_val, acc_retrain, acc_retrain_val

	


qbits_combs=[]
qbits_combs.append([32, 4, 8, 8, 4, 8, 8, 32])
#qbits_combs.append([32, 4, 8, 4, 8, 8, 8, 32])
#qbits_combs.append([32, 4, 8, 8, 4, 8, 8, 32])
#qbits_combs.append([32, 4, 8, 8, 8, 4, 8, 32])
#qbits_combs.append([32, 4, 8, 8, 8, 8, 4, 32])
#qbits_combs.append([32, 8, 4, 4, 8, 8, 8, 32])

# csv file initialization:
file_label = 'retrain_reward_acc'
headers = ['acc_val','acc_retrain', 'acc_retrain_val']
with open('./retrain_slops_files/'+file_label+'.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)

for i in range(len(qbits_combs)):
	qbits = qbits_combs[i]

	file_label = 'retrain_reward'
	for i in range(len(qbits)):
		file_label=file_label+'_'+str(qbits[i])

	# csv file initialization:
	headers = ['count', 'total', 'acc']
	with open('./retrain_slops_files/'+file_label+'.csv', 'w') as writeFile:
	    writer = csv.writer(writeFile)
	    writer.writerow(headers)

	network_name = 'alexnet'
	episode_num = 0
	layer_index = 1
	data = retrain(network_name, episode_num, layer_index, qbits)
	write_to_csv(data,'retrain_reward_acc')

