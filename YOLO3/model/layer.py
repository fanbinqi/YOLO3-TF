#coding:utf-8

import tensorflow as tf

def batch_normalization(input_data, input_channel, training, decay=0.9):
	'''
	:param input_data:  	输入数据，格式是(NHWC)
	:param input_channel: 	输入数据的维度
	:param training:  		是否进行训练，训练和测试的时候关于mean和variance的取法是不同的，训练时每一个batch更新一次，
							测试时根据滑动平均取值
	:param decay: 			均值方差滑动参数
	:return:
	'''
	with tf.variable_scope('batch_normalization'):
		scale = tf.get_variable(name='scale', shape=input_channel, dtype=tf.float32,
								initializer=tf.ones_initializer, trainable=True)
		shift = tf.get_variable(name='shift', shape=input_channel, dtype=tf.float32,
								initializer=tf.zeros_initializer, trainable=True)
		moving_mean = tf.get_variable(name='moving_mean', dtype=tf.float32, shape=input_channel,
								initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable(name='moving_var', dtype=tf.float32, shape=input_channel,
								initializer=tf.ones_initializer, trainable=False)

		def update_mean_var():
			axis = (0, 1, 2)
			batch_mean = tf.reduce_mean(input_data, axis=axis)  #这里对应了BN的三个维度：batch、H、W
			batch_var = tf.reduce_mean(tf.pow(input_data - batch_mean, 2), axis=axis)

			with tf.control_dependencies([tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay)),
										  tf.assign(moving_var, moving_var * decay + batch_var * (1 - decay))]):
				return tf.identity(moving_mean), tf.identity(moving_var)

		mean, variance = tf.cond(training, update_mean_var, lambda: (moving_mean, moving_var))
		return tf.nn.batch_normalization(input_data, mean, variance, shift, scale, 0.001)

def convolutional(name, input_data, filters_shape, training, downsample=False, activate=True, bn=True):
    """
    :param name: convolutional layer 的名字
    :param input_data: shape为(batch, height, width, channels)
    :param filters_shape: shape为(filter_height, filter_width, filter_channel, filter_num)
    :param training: 必须是tensor，True or False
    :param downsample: 是否对输入进行下采样
    :param activate: 是否使用激活函数
    :param bn: 是否使用batch normalization
    :return:
    """
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        if bn:
            conv = batch_normalization(input_data=conv, input_channel=filters_shape[-1], training=training)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate == True:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


def residual_block(name, input_data, input_channel, filter_num1, filter_num2, training):
    """
    :param name: residual_block的名字
    :param input_data: shape为(batch, height, width, channels)
    :param input_channel: input_data的channel
    :param filter_num1: residual block中第一个卷积层的卷积核个数
    :param filter_num2: residual block中第二个卷积层的卷积核个数
    :param training: 必须是tensor，True or False
    :return: residual block 的输出
    """
    with tf.variable_scope(name):
        conv = convolutional(name='conv1', input_data=input_data, filters_shape=(1, 1, input_channel, filter_num1),
                             training=training)
        conv = convolutional(name='conv2', input_data=conv, filters_shape=(3, 3, filter_num1, filter_num2),
                             training=training)
        residual_output = input_data + conv
    return residual_output

def pool(name, input_data, kernel_size = (1, 2, 2, 1), stride = (1, 2, 2, 1),
		 padding = 'SAME'):
	'''
	:param name:
	:param input_data: 			输入数据，格式为（NHWC）
	:param kernel_size: 		pooling的size，格式为[1, pooling_height, pooling_width, 1]
	:param stride:				pooling的stride，格式为[1, stride, stride, 1]
	:param padding:				是否padding 'SAME' or 'VALID'
	:return:
	'''
	with tf.variable_scope(name):
		pool_output = tf.nn.max_pool(value=input_data, ksize=kernel_size, strides=stride, padding=padding)
	return pool_output

def route(name, previous_output, current_output):
	'''
	:param name:  				由于维度信息是NHWC，所以axis=-1代表着是在channel的维度上进行concat的
	:param previous_output:		前面层的输出
	:param current_output:		当前层的输出
	:return:
	'''
	with tf.variable_scope(name):
		output = tf.concat([current_output, previous_output], axis=-1)
	return output

def upsample(name, input_data):
	'''
	:param name:
	:param input_data:    		输入数据shape为(batch, height, width, channels)
	:return:
	'''
	with tf.variable_scope(name):
		input_shape = tf.shape(input_data)
		output = tf.image.resize_bilinear(input_data, (input_shape[1] * 2, input_shape[2] * 2))
		#output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
		#print(output.shape)
	return output

def decode(name, conv_output, anchors, num_classes, stride):
	'''
	:param name:
	:param conv_output:  yolo的输出，shape为(batch_size, output_size, output_size, anchor_per_scale * (5 + num_classes))
	:param anchors:      存储格式为(w, h)，其中w, h的大小都是除了stride的
	:param num_classes:
	:param stride:
	:return: 			 shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes)
						 5 + num_classes指的是预测bbox的(x, y, w, h, confidence, probability)
						 其中(x, y, w, h)是预测bbox的中心坐标、宽、高，大小是相对于input_size的，
						 confidence是预测bbox属于物体的概率，probability是条件概率分布
	'''
	with tf.variable_scope(name):
		conv_shape = tf.shape(conv_output)
		batch_size = conv_shape[0]
		output_size = conv_shape[1]
		anchor_per_scale = len(anchors)

		conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
		output_xy = conv_output[:, :, :, :, 0:2]
		output_wh = conv_output[:, :, :, :, 2:4]
		output_conf = conv_output[:, :, :, :, 4:5]
		output_prob = conv_output[:, :, :, :, 5:]

		# 获取yolo的输出feature map中每个grid左上角的坐标
		# 需注意的是图像的坐标轴方向为
		#  - - - - > x
		# |
		# |
		# ↓
		# y
		# 在图像中标注坐标时通常用(y,x)，但此处为了与coor的存储格式(dx, dy, dw, dh)保持一致，将grid的坐标存储为(x, y)的形式
		y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
		x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
		xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
		xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
		xy_grid = tf.cast(xy_grid, tf.float32)
		# xy_grid = xy_grid + 0.5  SSD预测的是相对于中心店的偏移量，yolo预测的是相对于左上角的偏移量

		# (1)对xywh进行decode
		# dx，dy = sigmoid(output_xy)
		# dw, dh = exp(output_wh)
		# (x, y) = ((x_grid  y_grid) + (dx, dy)) * stride
		# (w, h) = ((anchor_w, anchor_h) * (w, h)) * stride
		pred_xy = (tf.sigmoid(output_xy) + xy_grid) * stride
		pred_wh = (tf.exp(output_wh) * anchors) * stride
		pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

		# (2)对confidence进行decode
		pred_conf = tf.sigmoid(output_conf)

		# (2)对probability进行decode
		pred_prob = tf.sigmoid(output_prob)

		pred_bbox = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

		return pred_bbox