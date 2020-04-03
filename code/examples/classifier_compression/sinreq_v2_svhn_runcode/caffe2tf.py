import sys

def process_oper(oper):
	name = oper[oper.rfind('=') + 2 : oper.rfind("'")]
	name = name.replace('-', '_')
	type = oper[1 : oper.find('(')]
	segments = oper[oper.find('(') + 1 : oper.find('name') - 1].replace(',', '').split(' ')
	params = []
	if type == 'conv':
		params = segments[3:]
	elif type == 'max_pool' or type == 'avg_pool':
		params = segments
	elif type == 'lrn':
		params = segments
	elif type == 'fc':
		params = segments[1:]
	elif type == 'concat':
		params = segments[0:]
	elif type == 'batch_normalization':
		params = segments
	return name, type, params

def process_file(infile_addr):
	oper_list = []
	oper_dict = {}
	with open(infile_addr, 'r') as infile:
		lines = infile.readlines()
		for i in range(0, len(lines)):
			lines[i] = lines[i].strip()
		i = 0
		counter = 0
		next_in = []
		while i < len(lines):
			line = lines[i]
			if line.startswith('(self.feed'):
				next_in = []
				next_in.append(line[line.find("'") + 1 : line.rfind("'")])
				while ')' not in line:
					i = i + 1
					line = lines[i]
					next_in.append(line[line.find("'") + 1 : line.rfind("'")])
			elif line.startswith('.'):
				name, type, params = process_oper(line)
				oper_dict[name] = counter
				counter = counter + 1
				oper_list.append([name, type, params, next_in])
				next_in = name
			i = i + 1
	return oper_list, oper_dict

def caffe2tf_noise(infile_addr, outfile_addr):
	oper_list, oper_dict = process_file(infile_addr)
	outfile = open(outfile_addr, 'w')
	for element in oper_list:
		name, type, params, inputs = element
		input_type = ''
		if type == 'conv':
			layer_def_1 = "namex = conv(input, weights_noisy['namex'], biases_noisy['namex'], params)\n".replace('namex', name)
			layer_def_2 = "err_lyr[\'namex\'] = tf.get_variable(name='namex_lyr_err', shape=namex.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])\n".replace('namex', name)
			layer_def_3 = "layers_err['namex'] = tf.add(namex, err_lyr['namex'])\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			if inputs == 'data':
				input_type == 'input'
			else:
				input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('input', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1 + layer_def_2 + layer_def_3)
		elif type == 'max_pool' or type == 'avg_pool':
			layer_def_1 = "namex = max_pool(input, params)\n".replace('namex', name)
			if type == 'avg_pool':
				layer_def_1 = layer_def_1.replace('max_pool', 'avg_pool')
			if isinstance(inputs, list):
				inputs = inputs[0]
			if inputs == 'data':
				input_type == 'input'
			else:
				input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('input', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'lrn':
			layer_def_1 = "namex = lrn(input, params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('input', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'fc':
			layer_def_1 = "namex = fc(input, weights_noisy['namex'], biases_noisy['namex'], params)\n".replace('namex', name)
			layer_def_2 = "err_lyr[\'namex\'] = tf.get_variable(name='namex_lyr_err', shape=namex.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])\n".replace('namex', name)
			layer_def_3 = "layers_err['namex'] = tf.add(namex, err_lyr['namex'])\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('input', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1 + layer_def_2 + layer_def_3)
		elif type == 'concat':
			layer_def_1 = "namex = concat([_xx_], params)\n".replace('namex', name)
			for i in range(len(inputs)):
				input_type = oper_list[oper_dict[inputs[i]]][1]
				if input_type == 'conv' or input_type == 'add':
					inputs[i] = "layers_err['" + inputs[i] + "']"
				layer_def_1 = layer_def_1.replace('_xx_', inputs[i] + ', _xx_')
			layer_def_1 = layer_def_1.replace(', _xx_', '')
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'add':
			layer_def_1 = "namex = add([_xx_])\n".replace('namex', name)
			layer_def_2 = "err_lyr[\'namex\'] = tf.get_variable(name='namex_lyr_err', shape=namex.shape[1:], initializer=tf.random_normal_initializer(mean=err_mean[3], stddev=err_stddev[3]), trainable=train_vars[3])\n".replace('namex', name)
			layer_def_3 = "layers_err['namex'] = tf.add(namex, err_lyr['namex'])\n".replace('namex', name)
			for i in range(len(inputs)):
				input_type = oper_list[oper_dict[inputs[i]]][1]
				if input_type == 'conv' or input_type == 'add':
					inputs[i] = "layers_err['" + inputs[i] + "']"
				layer_def_1 = layer_def_1.replace('_xx_', inputs[i] + ', _xx_')
			layer_def_1 = layer_def_1.replace(', _xx_', '')
			outfile.write(layer_def_1 + layer_def_2 + layer_def_3)
		elif type == 'relu':
			layer_def_1 = "namex = relu(input)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('input', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('input', inputs)
			outfile.write(layer_def_1)
		elif type == 'batch_normalization':
			layer_def_1 = "namex = batch_normalization(_input_, scale['namex'], offset['namex'], mean['namex'], variance['namex'], params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			if input_type == 'conv' or input_type == 'add':
				layer_def_1 = layer_def_1.replace('_input_', "layers_err['" + inputs + "']")
			else:
				layer_def_1 = layer_def_1.replace('_input_', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			layer_def_1 = layer_def_1.replace(', )', ')')
			outfile.write(layer_def_1)
	outfile.close()

def caffe2tf_noise_w(infile_addr, outfile_addr):
	oper_list, oper_dict = process_file(infile_addr)
	outfile = open(outfile_addr, 'w')
	for element in oper_list:
		name, type, params, inputs = element
		input_type = ''
		if type == 'conv':
			layer_def_1 = "namex = conv(input, weights_noisy['namex'], biases['namex'], params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			if inputs == 'data':
				input_type == 'input'
			else:
				input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'max_pool' or type == 'avg_pool':
			layer_def_1 = "namex = max_pool(input, params)\n".replace('namex', name)
			if type == 'avg_pool':
				layer_def_1 = layer_def_1.replace('max_pool', 'avg_pool')
			if isinstance(inputs, list):
				inputs = inputs[0]
			if inputs == 'data':
				input_type == 'input'
			else:
				input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'lrn':
			layer_def_1 = "namex = lrn(input, params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'fc':
			layer_def_1 = "namex = fc(input, weights_noisy['namex'], biases['namex'], params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('input', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'concat':
			layer_def_1 = "namex = concat([_xx_], params)\n".replace('namex', name)
			for i in range(len(inputs)):
				input_type = oper_list[oper_dict[inputs[i]]][1]
				layer_def_1 = layer_def_1.replace('_xx_', inputs[i] + ', _xx_')
			layer_def_1 = layer_def_1.replace(', _xx_', '')
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
		elif type == 'add':
			layer_def_1 = "namex = add([_xx_])\n".replace('namex', name)
			for i in range(len(inputs)):
				input_type = oper_list[oper_dict[inputs[i]]][1]
				layer_def_1 = layer_def_1.replace('_xx_', inputs[i] + ', _xx_')
			layer_def_1 = layer_def_1.replace(', _xx_', '')
			outfile.write(layer_def_1)
		elif type == 'relu':
			layer_def_1 = "namex = relu(input)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('input', inputs)
			outfile.write(layer_def_1)
		elif type == 'batch_normalization':
			layer_def_1 = "namex = batch_normalization(_input_, scale['namex'], offset['namex'], mean['namex'], variance['namex'], params)\n".replace('namex', name)
			if isinstance(inputs, list):
				inputs = inputs[0]
			input_type = oper_list[oper_dict[inputs]][1]
			layer_def_1 = layer_def_1.replace('_input_', inputs)
			for param in params:
				layer_def_1 = layer_def_1.replace('params', param + ', params')
			layer_def_1 = layer_def_1.replace(', params', '')
			outfile.write(layer_def_1)
	outfile.close()
	
#caffe2tf_noise(sys.argv[1], sys.argv[2])
caffe2tf_noise_w(sys.argv[1], sys.argv[2])