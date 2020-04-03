import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import scipy
import pickle
#font = {'family': 'Times', 'size': 20}
#matplotlib.rc('font', **font)
import load

def get_best_fit(data, x, y):
	mse_min = sys.maxsize
	#dist_list = [stats.uniform, stats.norm, stats.cauchy, stats.laplace, stats.johnsonsu]
	#dist_names = ['uniform', 'norm', 'cauchy', 'laplace', 'johnsonsu']
	dist_list = [stats.cauchy]
	dist_names = ['cauchy']
	for i in range(len(dist_list)):
		dist = dist_list[i]
		params = dist.fit(data)
		param_opt, _ = curve_fit(dist.pdf, x, y, scipy.array(params))
		mse = ((dist.pdf(x, *param_opt) - y) ** 2).mean()
		if mse < mse_min:
			best_dist = dist
			best_params = params
			mse_min = mse
			best_name = dist_names[i]
	return best_dist, best_params, best_name

def plot_dist(path_params, path_errors, save_name):
	#path_params = '/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/lenet/lenet.ckpt'
	#path_errors = '/home/behnam/results/lenet/'
	weights = load.get_netparams(path_params)
	errors = load.get_error(path_errors)

	n_row = len(weights[0])
	n_col = 3
	count = 1
	n_bins = 200
	f = plt.figure(figsize=(n_row*3, n_row*3))
	for layer in weights[0]:
		print(layer)
		sub_weights = load.get_subparams(weights, [layer])
		sub_errors = load.get_suberror(errors, 'w', [layer])
		y, x = np.histogram(sub_weights, bins=n_bins, normed=True)
		ax = f.add_subplot(n_row, n_col, count)
		ax.set_xlim([-0.5, 0.5])
		x = x[:-1]
		scale = 1 / max(y)
		ax.hist(x, len(x), weights=scale*y, color='skyblue')
		best_dist, best_params, best_name = get_best_fit(sub_weights, x, y)
		label = layer + '['
		for n in best_params:
			label = label + '%.3f ' % n
		label = label + ']'
		ax.plot(x, scale * best_dist.pdf(x, *best_params), color='black', label=label)
		plt.legend(loc='upper left')
		
		y, x = np.histogram(sub_errors, bins=n_bins, normed=True)
		ax = f.add_subplot(n_row, n_col, count+1)
		ax.set_xlim([-0.5, 0.5])
		x = x[:-1]
		scale = 1 / max(y)
		ax.hist(x, len(x), weights=scale*y, color='salmon')
		best_dist, best_params, best_name = get_best_fit(sub_errors, x, y)
		label = '['
		for n in best_params:
			label = label + '%.3f ' % n
		label = label + ']'
		ax.plot(x, scale * best_dist.pdf(x, *best_params), color='black', label=label)
		plt.legend(loc='upper left')
		
		ax = f.add_subplot(n_row, n_col, count+2)
		ax.set_xlim([-0.1, 0.1])
		ax.hist(x, len(x), weights=scale*y, color='salmon')
		ax.plot(x, scale * best_dist.pdf(x, *best_params), color='black')
		count = count + 3
	for name in save_name:
		plt.savefig(save_name)
'''
#def plot_weights_dist
with open('/home/behnam/results/deltas/lenet.pickle', 'r') as f:
	weights_noisy_array = pickle.load(f)
for layer in weights_noisy_array[0]:
	print(layer)
	#for i in range (0,len(weights_noisy_array)):
	w_dist_ip1 = []
	w_dist_conv1 = []
	w_dist_conv2 = []
	w_dist_ip2 = []
	for i in range (0,len(weights_noisy_array)):
		w_dist_ip1.append((weights_noisy_array[i]['ip1'][1][1]))
		w_dist_conv1.append((weights_noisy_array[i]['conv1'][1][1][0][1]))
		w_dist_conv2.append((weights_noisy_array[i]['conv2'][1][1][0][1]))
		w_dist_ip2.append((weights_noisy_array[i]['ip2'][1][1]))
		weights_noisy_array[i]['conv2'].ravel()

	print(len(w_dist_ip1))
	print(len(w_dist_conv1))
	print(len(w_dist_conv2))
	print(len(w_dist_ip2))

# plotting the weights distribution over 100 run 
import seaborn as sns 
sns.set()
_ = plt.hist(w_dist_ip1)
_ = plt.xlabel('Network = LeNET - Layer = ip1 - weight [ip1][1][1]')
_ = plt.ylabel('Frequency')
plt.savefig('/home/behnam/results/deltas/1.png')
sns.set()
_ = plt.hist(w_dist_ip2)
_ = plt.xlabel('Network = LeNET - Layer = ip2 - weight [ip2][1][1]')
_ = plt.ylabel('Frequency')
plt.savefig('/home/behnam/results/deltas/w2.png')
sns.set()
_ = plt.hist(w_dist_conv1)
_ = plt.xlabel('Network = LeNET - Layer = conv1 - weight [conv1][1][1][0][1]')
_ = plt.ylabel('Frequency')
plt.savefig('/home/behnam/results/deltas/w3.png')
sns.set()
_ = plt.hist(w_dist_conv2)
_ = plt.xlabel('Network = LeNET - Layer = conv2 - weight [conv2][1][1][0][1]')
_ = plt.ylabel('Frequency')
plt.savefig('/home/behnam/results/deltas/w4.png')
#for layer in weights_noisy_array[0]:
#	weights_noisy_array[:][layer]
'''

'''
netparams = load.get_netparams('/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt')
weights = netparams['weights']
weights_layer = weights['conv5'].ravel()
weights_layer.sort()
weights_layer_diff = np.diff(weights_layer)

weights_layer_diff_set = list(set(weights_layer_diff))
'''

'''
# quantization
netparams = load.get_netparams('/home/behnam/rlbitwidth.tfmodels/caffe2tf/tfmodels/alexnet/alexnet.ckpt')
weights = netparams['weights']
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for layer in layers:
	weights_layer = weights[layer].ravel()
	print(layer + '\t' + (str)(np.var(weights_layer)))
'''

'''
errors = load.get_error('/home/behnam/results/networks/alexnet/480.0_81.2092')
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for layer in layers:
	sub_errors = load.get_suberror(errors, 'w', [layer])
	print(layer + '\t' + (str)(np.var(sub_errors.ravel())))
'''