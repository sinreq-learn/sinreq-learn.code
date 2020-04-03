# start

def eval_imagenet(net_name, param_path, param_q_path, qbits, layer_index, layer_name, file_idx, shift_back, trainable=False, err_mean=None, err_stddev=None, train_vars=None, cost_factor=200., n_epoch=1):
	"""all layers are trainable in the conventional retraining procedure"""
	if '.ckpt' in param_path:
		netparams = load.load_netparams_tf(param_path, trainable=True)
	else:
		netparams = load.load_netparams_tf_q(param_path, trainable=True)

	data_spec = helper.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.int32)

	elif net_name == 'alexnet':
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
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) + cost_factor*q_diff_cost

	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=label_node)) 

	probs  = helper.softmax(logits_)
	top_k_op = tf.nn.in_top_k(probs, label_node, 5)
	#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.1)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=10-8)
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
				path_train = '/home/ahmed/projects/NN_quant/imageNet_training'

				image_producer = dataset.ImageNetProducer(val_path=path_train + '/train_shuf_'+str(file_idx)+'.txt', data_path=path_train, data_spec=data_spec)
				#image_producer = dataset.ImageNetProducer(val_path=path_train + '/train_shuf_100images.txt', data_path=path_train, data_spec=data_spec)
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
					top, weights_conv4_tmp_ret = sess.run([top_k_op, weights_conv4_tmp], feed_dict={input_node: images, label_node: labels})
					correct += np.sum(top)
					#print(np.amax(weights_q_tmp))
					#print(len(set(weights_q_tmp.ravel())))
					# --------
					count += len(labels)
					cur_accuracy = float(correct) * 100 / count
					write_to_csv([count, total, cur_accuracy])
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
			path_val = './nn_quant_and_run_code_train/ILSVRC2012_img_val'
			image_producer = dataset.ImageNetProducer(val_path=path_val + '/val_1k.txt', data_path=path_val, data_spec=data_spec)
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


