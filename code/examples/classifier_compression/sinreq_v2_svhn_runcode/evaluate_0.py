import numpy as np
import tensorflow as tf
import pickle
import os
import dataset
import networks
from networks import *
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
import load
import json

#NETWORKS = ['alexnet', 'googlenet', 'nin', 'resnet18', 'resnet50', 'squeezenet', 'vgg16net', 'lenet']
NETWORKS = ['lenet']
IMAGE_PATH_TRAIN = '/home/behnam/ILSVRC2012_img_val_40K/'
IMAGE_PATH_TEST = '/home/behnam/ILSVRC2012_img_val_10K/'
IMAGE_LABLE = '/home/behnam/rlbitwidth.code/val.txt'
CKPT_PATH = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/'
NET_ACC = {'alexnet': 79.918, 'googlenet': 89.002, 'nin': 81.218, 'resnet18': 85.016, 
	'resnet50': 91.984, 'squeezenet': 80.346, 'vgg16net': 89.816, 'lenet': 99.06}


def eval_imagenet(net_name, param_path, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=200., n_epoch=1):
	if '.ckpt' in param_path:
		netparams = load.load_netparams_tf(param_path, trainable=True)
	else:
		netparams = load.load_netparams_tf_q(param_path, trainable=True)
	#print((len(netparams['biases'])))
	data_spec = networks.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.int32)
	if net_name == 'alexnet_noisy':
		logits_, err_w, err_b, err_lyr = networks.alexnet_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'alexnet':
		logits_ = networks.alexnet(input_node, netparams)
	elif net_name == 'alexnet_shift':
		logits_ = networks.alexnet_shift(input_node, netparams)
	elif net_name == 'googlenet':
		logits_, err_w, err_b, err_lyr = networks.googlenet_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'nin':
		logits_, err_w, err_b, err_lyr = networks.nin_noisy(input_node, netparams, err_mean, err_stddev, train_vars)
	elif net_name == 'resnet18':
		logits_ = networks.resnet18(input_node, netparams)
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
	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor / (1. + square_sum)
	
	# ======== calculating the quantization error of a certain layer
	#with open('/home/behnam/results/quantized/alexnet/alexnet_8.pickle', 'r') as f:
	#	params_8 = pickle.load(f)

	'''
	Wo_bar_q = '/home/behnam/results/weights_retrained/alexnet_conv2_7.pickle' # save = {Wo}'q (quantized version)
	with open('/home/behnam/results/weights_retrained/alexnet_conv2_7.pickle', 'r') as f:
		params_quantized = pickle.load(f)
	layer = 'conv2'
	params_quantized_layer = tf.get_variable(name='params_quantized_layer', initializer=tf.constant(params_quantized[0][layer]), trainable=False)
	'''
	#q_diff = tf.subtract(params_quantized_layer, netparams['weights'][layer])
	#q_diff_cost = tf.nn.l2_loss(q_diff)
	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor*q_diff_cost
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) 
	
	probs  = softmax(logits_)
	top_k_op = tf.nn.in_top_k(probs, label_node, 5)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.1)
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
				image_producer = dataset.ImageNetProducer(val_path='/home/behnam/ILSVRC2012_img_val_40K/val_40.txt', data_path='/home/behnam/ILSVRC2012_img_val_40K', data_spec=data_spec)
				total = len(image_producer) * n_epoch
				coordinator = tf.train.Coordinator()
				threads = image_producer.start(session=sess, coordinator=coordinator)
				for (labels, images) in image_producer.batches(sess):
					one_hot_labels = np.zeros((len(labels), 1000))
					for k in range(len(labels)):
						one_hot_labels[k][labels[k]] = 1
					sess.run(train_op, feed_dict={input_node: images, label_node: one_hot_labels})
					correct += np.sum(sess.run(top_k_op, feed_dict={input_node: images, label_node: labels}))
					count += len(labels)
					cur_accuracy = float(correct) * 100 / count
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
			#image_producer = dataset.ImageNetProducer(val_path='/home/behnam/ILSVRC2012_img_val_10K/val_10.txt', data_path='/home/behnam/ILSVRC2012_img_val_10K', data_spec=data_spec)				image_producer = dataset.ImageNetProducer(val_path='/home/behnam/ILSVRC2012_img_val_40K/val_40.txt', data_path='/home/behnam/ILSVRC2012_img_val_40K', data_spec=data_spec)
			image_producer = dataset.ImageNetProducer(val_path='/home/behnam/ILSVRC2012_img_val_40K/val_40.txt', data_path='/home/behnam/ILSVRC2012_img_val_40K', data_spec=data_spec)
			total = len(image_producer)
			coordinator = tf.train.Coordinator()
			threads = image_producer.start(session=sess, coordinator=coordinator)
			for (labels, images) in image_producer.batches(sess):
				one_hot_labels = np.zeros((len(labels), 1000))
				for k in range(len(labels)):
					one_hot_labels[k][labels[k]] = 1
				correct += np.sum(sess.run(top_k_op, feed_dict={input_node: images, label_node: labels}))
				count += len(labels)
				cur_accuracy = float(correct) * 100 / count
				print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
			coordinator.request_stop()
			coordinator.join(threads, stop_grace_period_secs=2)
			return cur_accuracy, 0

def eval_imagenet_q(net_name, param_pickle_path):
	netparams = load.load_netparams_tf_q(param_pickle_path)
	data_spec = networks.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.int32)
	if net_name == 'alexnet':
		logits_ = networks.alexnet(input_node, netparams)
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
			#print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
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
		directory = '/home/behnam/results/networks/' + net_name
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
	directory = '/home/behnam/results/deltas/' + net_name
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
path_net = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py'
layers = load.get_layers(path_net)
acc = {}
for i in range(0, len(layers)):
	path = '/home/behnam/results/quantized/resnet18/resnet18_10_' + layers[i] + '_7.pickle'
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
	for dirpath, subdirs, fileList in os.walk('/home/behnam/results/normalized/resnet18/'):
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
	


#save_path_params = '/home/behnam/results/weights_retrained/alexnet_conv2_retrained_test.pickle' # {Wo}'
save_path_params = '/home/behnam/results/weights_retrained/alexnet_conv2_retrained_test_f.pickle' # = {W1}

Wo = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt'
Wo_bar_q = '/home/behnam/results/weights_retrained/alexnet_conv2_7.pickle' # save = {Wo}'q (quantized version) - then ll be used to check testing acc 
W1_q = '/home/behnam/results/weights_retrained/alexnet_conv2_7_W1q.pickle'

Wo_bar = '/home/behnam/results/weights_retrained/alexnet_conv2_retrained_test.pickle' # = {Wo}'
W1 = '/home/behnam/results/weights_retrained/alexnet_conv2_retrained_test_f.pickle' # = {W1}
test = '/home/behnam/results/weights_retrained/test.pickle' # test = re-training using latest layers (fcs)


# ===== FOR TRAINING ====================
'''
# this is for phase I training - retrain a little bit on new dataset 40K - get Wo'
#param_path = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt' # = {Wo}
# this is for phase II training - retrain to minimize the quantization error - >> get W1

#input_file = Wo_bar
input_file = Wo
#output_file = W1
output_file = test

param_path = input_file 
save_path_params = output_file 
acc, netparams = eval_imagenet('alexnet', param_path, trainable=True, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)
print(acc)
with open(save_path_params, 'w') as f:
	pickle.dump(netparams, f)
'''

# ===== FOR INFERENCE: RESNET18 ====================
#'''

#param_path = '/home/behnam/results/normalized/resnet18/res2a_branch2a3_shift_04May.pickle'
#param_path = '/home/behnam/results/quantized/resnet18/resnet18_layers_shift_and_quant.pickle'
path_save = '/home/behnam/results/quantized/resnet18/resnet18'
path_save_q = path_save + '_layers_shift_quant_05May.pickle'
param_path = path_save_q
#param_path = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.ckpt'
#param_path = '/home/behnam/results/quantized/resnet18/resnet18_10.pickle'
shift_back = {}
layers_sorted = load.get_layers('/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/resnet18/resnet18.py')
for layer in layers_sorted:
	shift_back[layer] = 0

shift_back['res2a_branch1'] = 1 # 84.69
#shift_back['res2a_branch2a'] =  # x
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




acc, netparams = eval_imagenet('resnet18_shift', param_path, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)
print(acc)
#'''


# ===== FOR INFERENCE: ALEXNET ====================
'''
#Wo_bar_q = '/home/behnam/results/weights_retrained/alexnet_conv2_7.pickle' # save {Wo}'q (quantized version) - then ll be used to check testing acc 
uniform_5bits = '/home/behnam/results/weights_retrained/alexnet_W_muLAW_uniform.pickle'
one_5bits = '/home/behnam/results/weights_retrained/alexnet_W_muLAW_one.pickle'
norm_5bits = '/home/behnam/results/weights_retrained/alexnet_W_muLAW_norm.pickle'

#param_path = '/home/behnam/results/normalized/resnet18/res2a_branch2a3_shift_04May.pickle'
param_path = Wo
acc, netparams = eval_imagenet('alexnet', param_path, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=0., n_epoch=1)
print(acc)
'''
