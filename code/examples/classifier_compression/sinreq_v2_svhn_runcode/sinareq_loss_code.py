# SinAReQ - loss - Code
def train_with_sinareq(net_name, qbits_dict={}, layer_index=[], layer_name=[], trainable=False, train_vars=None, cost_factor=0., n_epoch=10, init=0):
	ckpt_path = 'PATH/latest.ckpt'

	data_spec = helper.get_data_spec(net_name)
	input_node = tf.placeholder(tf.float32, shape=(None, data_spec.crop_size * data_spec.crop_size * data_spec.channels))
	input_node_2d = tf.reshape(input_node, shape=(-1, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
	label_node = tf.placeholder(tf.float32, [None, 10])

	if trainable:
		num_steps_to_show_loss = 100
		num_steps_to_check = 1000    
		with tf.Graph().as_default():
			if init:
			    netparams = load.init_svhn_netparams_tf(ckpt_path, trainable=True)
			else:
			    netparams = load.load_svhn_netparams_tf(ckpt_path, trainable=True)
			print('loading checkpoint model params ..')
			path_to_train_tfrecords_file = 'PATH/data/val.tfrecords'
			batch_size = batch_size_train

			image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_train_tfrecords_file,
			                                                             num_examples=num_train_examples,
			                                                             batch_size=batch_size,
			                                                             shuffled=True)
			
			""" forward pass """ 
			length_logits, digits_logits = svhn_net.svhn_net(image_batch, netparams)
			#length_logits, digits_logits = svhn_net.svhn_net_q(image_batch, netparams, qbits_dict)

			""" 1) Initialization: Bitwidth regularization (i.e., learning the sinusoidal period) """
			# num_bits = \beta in the paper formulation 
			num_bits_1 = tf.get_variable(name="freq", initializer=8.0, trainable=True)
			num_bits_2 = tf.get_variable(name="freq_2", initializer=8.0, trainable=True)
			num_bits_3 = tf.get_variable(name="freq_3", initializer=8.0, trainable=True)
			num_bits_4 = tf.get_variable(name="freq_4", initializer=8.0, trainable=True)
			num_bits_5 = tf.get_variable(name="freq_5", initializer=8.0, trainable=True)
			num_bits_6 = tf.get_variable(name="freq_6", initializer=8.0, trainable=True)

			""" 2) Initialization: Weight quantization regularization """
			sin2_loss_1 = tf.constant(0.0)
			sin2_loss_2 = tf.constant(0.0)
			sin2_loss_3 = tf.constant(0.0)
			sin2_loss_4 = tf.constant(0.0)
			sin2_loss_5 = tf.constant(0.0)
			sin2_loss_6 = tf.constant(0.0)

			""" 3) Initialization: Hyperparameters + losses"""
			lambda_q = tf.constant(0.0)
			lambda_f = tf.constant(0.0)
			freq_loss = tf.constant(0.0)
			sin2_loss = tf.constant(0.0)

			layer_name = 'layer_1_name'
			qbits = qbits_dict[layer_name]
				#sin2_func_1 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights']['conv2']/(2**(-(qbits[1]-1))))))
				#sin2_func_0 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-(qbits))))))
				#sin2_func_1 = tf.reduce_mean(tf.square(tf.sin(pi*(netparams['weights']['conv2']+2**-(qbits[1]))/(2**(-(qbits[1]-1))))))
				" R0(w,b)" 
				#sin2_loss = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits)))))) 
				" R1(w,b)" 
				sin2_loss = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits)-1)))/2**(num_bits)) 
				" R2(w,b)" 
				#sin2_loss = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits)-1)))/4**(num_bits)) 
				
			layer_name = 'layer_2_name'
			qbits = qbits_dict[layer_name]
				#sin2_func_1 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights']['conv2']/(2**(-(qbits[1]-1))))))
				#sin2_func_1 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-(qbits))))))
				#sin2_func_1 = tf.reduce_mean(tf.square(tf.sin(pi*(netparams['weights']['conv2']+2**-(qbits[1]))/(2**(-(qbits[1]-1))))))

				#sin2_loss = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-(num_bits)))))) 
				" R0(w,b)" 
				#sin2_loss_2 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits_2)))))) 
				" R1(w,b)" 
				sin2_loss_2 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_2)-1)))/2**(num_bits_2)) 
				" R2(w,b)" 
				
			layer_name = 'layer_3_name'
			qbits = qbits_dict[layer_name]
				" R0(w,b)" 
				#sin2_loss_3 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits_3)))))) 
				" R1(w,b)" 
				sin2_loss_3 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_3)-1)))/2**(num_bits_3)) 
				" R2(w,b)" 
				#sin2_loss_3 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_3)-1)))/4**(num_bits_3)) 
				freq_loss_3 = num_bits_3
				
			layer_name = 'layer_4_name'
			qbits = qbits_dict[layer_name]
				" R0(w,b)" 
				#sin2_loss_4 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits_4)))))) 
				" R1(w,b)" 
				sin2_loss_4 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_4)-1)))/2**(num_bits_4)) 
				" R2(w,b)" 
				#sin2_loss_4 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_4)-1)))/4**(num_bits_4)) 
				
			layer_name = 'layer_5_name'
			qbits = qbits_dict[layer_name]
				" R0(w,b)" 
				#sin2_loss_5 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits_5)))))) 
				" R1(w,b)" 
				sin2_loss_5 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_5)-1)))/2**(num_bits_5)) 
				" R2(w,b)" 
				#sin2_loss_5 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_5)-1)))/4**(num_bits_5)) 
	
			layer_name = 'layer_6_name'
			qbits = qbits_dict[layer_name]
				" R0(w,b)" 
				#sin2_loss_6 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]/(2**(-tf.identity(num_bits_6)))))) 
				" R1(w,b)" 
				sin2_loss_6 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_6)-1)))/2**(num_bits_6)) 
				" R2(w,b)" 
				#sin2_loss_6 = tf.reduce_mean(tf.square(tf.sin(pi*netparams['weights'][layer_name]*(2**(num_bits_6)-1)))/4**(num_bits_6)) 

			""" ------------------------------------------------ """

			""" loss calculation """ 
			loss_sin2_reg = lambda_q * cost_factor * (sin2_loss + sin2_loss_2 + sin2_loss_3 + sin2_loss_4 + sin2_loss_5 + sin2_loss_6) 
			loss_freq_reg = lambda_f * 1 * (num_bits + num_bits_2 + num_bits_3 + num_bits_4 + num_bits_5 + num_bits_6)

			#loss_op = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy + loss_sin2_reg + loss_freq_reg 
			acc_loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy 
			loss_op = acc_loss + loss_sin2_reg + loss_freq_reg 

			global_step = tf.Variable(0, name='global_step', trainable=False)
			training_options = {}
			training_options['learning_rate'] =  1e-2
			training_options['decay_steps'] =  10000
			training_options['decay_rate'] = 0.9
			learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(loss_op, global_step=global_step)
			
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
				saver = tf.train.Saver()

				
				print('=> Start training ..')
				print('########################################')

				#cur_accuracy = 0
				#patience = initial_patience
				best_accuracy = 0.0
				duration = 0.0

				for i in range(0, n_epoch): 
					step = i
					n_steps = n_epoch
					
					#print(' debug # 0')
					" Method 0 "
					#lambda_q_value = (1/2)*np.exp(i*2/n_epoch) # rising1
					#lambda_f_value = (1/2)*np.exp(i*2/n_epoch) # rising1
					#lambda_f_value = 0.01

					"Method 2:  step-like lambda "
					r = 0.2*n_epoch
					d = 0.8*n_epoch
					s = 20
					f1 = 0.5 * (1+np.tanh((i-r)/s));
					f2 = 0.5 * (1+np.tanh((i-d)/s));
					lambda_q_value = f1
					#lambda_f_value = 0.02*(f1-f2)
					lambda_f_value = 0.03

					#"Method 1:  old method  "
					#scale = 3
					#shift = 2
					#lambda_q_value = 0.5*(np.tanh(pi*scale*(step-n_steps/shift)/n_steps)-np.tanh(pi*scale*(0-n_steps/shift)/n_steps));
					#scale = 3
					#shift = 2
					#lambda_f_value  = 1/(np.cosh(pi*scale*(step-n_steps/shift)/n_steps))
					
					start_time = time.time()
					#_, loss_val, global_step_val = sess.run([train_op, loss_op, global_step])
					_, loss_val, acc_loss_val, loss_sin2_reg_val, loss_freq_reg_val, global_step_val = sess.run([train_op, loss_op, acc_loss, loss_sin2_reg, loss_freq_reg, global_step], feed_dict={lambda_q: lambda_q_value, lambda_f: lambda_f_value})
					sin2_l, sin2_l_2,  sin2_l_3, sin2_l_4, sin2_l_5, sin2_l_6 = sess.run([sin2_loss, sin2_loss_2,  sin2_loss_3, sin2_loss_4, sin2_loss_5, sin2_loss_6]) 
					n_bits , n_bits_2 , n_bits_3 , n_bits_4 , n_bits_5 , n_bits_6 = sess.run([num_bits , num_bits_2 , num_bits_3 , num_bits_4 , num_bits_5 , num_bits_6])
					duration += time.time() - start_time
					
					#print('=> %s: step %d, loss = %f ' % (
					#    	datetime.now(), global_step_val, loss_val))

					print('=> %s: step %d, total_loss = %f, sin2_reg_loss = %f , freq_reg_loss = %f' % (
					    	datetime.now(), global_step_val, loss_val, loss_sin2_reg_val, loss_freq_reg_val))
					print("lambda_q_value = ", lambda_q_value)
					print('=> l1 = %f , l2 = %f , l3 = %f , l4 = %f , l5 = %f , l6 = %f' % (n_bits , n_bits_2 , n_bits_3 , n_bits_4 , n_bits_5 , n_bits_6))
					
					"""
					if global_step_val % num_steps_to_show_loss == 0:
						examples_per_sec = batch_size * num_steps_to_show_loss / duration
						duration = 0.0
						print('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
					    	datetime.now(), global_step_val, loss_val, examples_per_sec))

					if global_step_val % num_steps_to_check != 0:
						continue
					"""
					#_, loss_val = sess.run([train_op, loss_op])
					write_to_data([lambda_q_value, lambda_f_value, loss_val, acc_loss_val, loss_sin2_reg_val, loss_freq_reg_val, \
							sin2_l, sin2_l_2,  sin2_l_3, sin2_l_4, sin2_l_5, sin2_l_6, \
							 n_bits, n_bits_2, n_bits_3, n_bits_4, n_bits_5, n_bits_6])
					print('---------------- finished epoch# ', i)
				
					netparams_save  = sess.run(netparams)
					print(' Training finished')

					""" path for saving the retrained model """
					network_name = 'svhn_net'
					path_save = './results_retrained_models/' + network_name + '/quantized/' + network_name
					path_save_params = path_save + '_retrained_lambda_'+str(cost_factor)+'_CONV.pickle'
					# AHMED: debug
					#print('retrained = ', np.amax(netparams['weights']['conv2']))
					#print('len set = ', len(set(np.array(netparams['weights']['conv2']))))
					# ------------
					#print('=================================================')
				print('=> Writing trained model parameters ...')
				#print('=================================================')
				print(len(netparams_save['weights']))
				print(netparams_save['weights'].keys())
				with open(path_save_params, 'wb') as f:
					pickle.dump(netparams_save, f)

				print('=> Evaluating on validation dataset...')
				accuracy_val = evaluator_svhn(path_save_params, qbits_dict)
				print('epoch #', i)
				print('accuracy', accuracy_val)
				coord.request_stop()
				coord.join(threads)	
				return accuracy_val	

