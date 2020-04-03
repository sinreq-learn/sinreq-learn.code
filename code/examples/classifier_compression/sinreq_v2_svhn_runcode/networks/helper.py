import numpy as np
import tensorflow as tf

def conv(input, kernel, biases, s_h, s_w, relu=True, padding='SAME', group=1, biased=True):
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	if group == 1:
		output = convolve(input, kernel)
	else:
		input_groups = tf.split(input, group, 3)
		kernel_groups = tf.split(kernel, group, 3)
		output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
		output = tf.concat(output_groups, 3)
	if biased:
		output = tf.nn.bias_add(output, biases)
	if relu:
		output = tf.nn.relu(output)
	return output
	
def max_pool(input, k_h, k_w, s_h, s_w, padding='SAME'):
	return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
	
def avg_pool(input, k_h, k_w, s_h, s_w, padding='SAME'):
	return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
	
def softmax(input):
	input_shape = list(map(lambda v: v.value, input.get_shape()))
	#input_shape = map(lambda v: v.value, input.get_shape())
	if len(list(input_shape)) > 2:
		if input_shape[1] == 1 and input_shape[2] == 1:
			input = tf.squeeze(input, squeeze_dims=[1, 2])
		else:
			raise ValueError('Rank 2 tensor input expected for softmax!')
	return tf.nn.softmax(input)
	
def lrn(input, radius, alpha, beta, bias=1.0):
	return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def fc(input, weights, biases, relu=True):
	input_shape = input.get_shape()
	if input_shape.ndims == 4:
		dim = 1
		for d in input_shape[1:].as_list():
			dim *= d
		feed_in = tf.reshape(input, [-1, dim])
	else:
		feed_in, dim = (input, input_shape[-1].value)
	op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
	return op(feed_in, weights, biases)

def concat(inputs, axis):
	return tf.concat(axis=axis, values=inputs)

def add(inputs):
	#return tf.add(inputs[0], inputs[1])
	return tf.add_n(inputs)
		
def relu(input):
	return tf.nn.relu(input)
		
def make_var(name, shape, trainable=True):
	return tf.get_variable(name, shape, trainable=trainable)
		
def batch_normalization(input, scale, offset, mean, variance, scale_offset=True, relu=False):
	if scale_offset:
		scale = scale
		offset = offset
	else:
		scale, offset = (None, None)
	output = tf.nn.batch_normalization(
		input,
		mean=mean,
		variance=variance,
		offset=offset,
		scale=scale,
		# TODO: This is the default Caffe batch norm eps
		# Get the actual eps from parameters
		variance_epsilon=1e-5)
	if relu:
		output = tf.nn.relu(output)
	return output

		
#----- Classes and methods required to get network data specification, e.g., batch size, crop size, etc. -----#
class DataSpec(object):
	def __init__(self, batch_size, scale_size, crop_size, isotropic, channels=3, mean=None, bgr=True):
			self.batch_size = batch_size
			self.scale_size = scale_size
			self.isotropic = isotropic
			self.crop_size = crop_size
			self.channels = channels
			self.mean = mean if mean is not None else np.array([104., 117., 124.])
			self.expects_bgr = True
	
def alexnet_spec(batch_size=20):
	return DataSpec(batch_size=batch_size, scale_size=256, crop_size=227, isotropic=False)
	
def svhn_net_spec(batch_size=128):
	return DataSpec(batch_size=batch_size, scale_size=32, crop_size=32, isotropic=False)

def lenet_spec(batch_size=128):
	return DataSpec(batch_size=batch_size, scale_size=28, crop_size=28, isotropic=False, channels=1)
	
def std_spec(batch_size, isotropic=True):
	return DataSpec(batch_size=batch_size, scale_size=256, crop_size=224, isotropic=isotropic)

MODEL_DATA_SPECS = {
	'svhn_net': svhn_net_spec(),
	'alexnet': alexnet_spec(),
	'squeezenet': alexnet_spec(),
	'caffenet': alexnet_spec(),
	'googlenet': std_spec(batch_size=20, isotropic=False),
	'resnet18': std_spec(batch_size=25),
	'resnet18_noisy': std_spec(batch_size=25),
	'resnet18_shift': std_spec(batch_size=25),
	'resnet50': std_spec(batch_size=25),
	'resnet101': std_spec(batch_size=25),
	'resnet152': std_spec(batch_size=25),
	'nin': std_spec(batch_size=20),
	'vgg16net': std_spec(batch_size=100),
	'lenet': lenet_spec()
}
	
def get_data_spec(model_class):
	return MODEL_DATA_SPECS[model_class]

def add_noise(weights, biases, err_mean, err_stddev, train_vars):
	err_w = {}
	err_b = {}
	weights_noisy = {}
	biases_noisy = {}
	for layer in weights:
		err_w[layer] = tf.get_variable(name=layer+'_w_err', shape=weights[layer].shape, initializer=tf.random_normal_initializer(mean=err_mean[1], stddev=err_stddev[1]), trainable=train_vars[1])
		weights_noisy[layer] = tf.add(err_w[layer], weights[layer])
		if layer in biases:
			err_b[layer] = tf.get_variable(name=layer+'_b_err', shape=biases[layer].shape,  initializer=tf.random_normal_initializer(mean=err_mean[2], stddev=err_stddev[2]), trainable=train_vars[2])
			biases_noisy[layer]  = tf.add(err_b[layer], biases[layer])
		else:
			biases_noisy[layer] = tf.get_variable(name=layer, shape=weights[layer].shape[-1], initializer=tf.zeros_initializer, trainable=False)
	return weights_noisy, biases_noisy, err_w, err_b
