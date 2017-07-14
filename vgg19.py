import tensorflow as tf

# VGG19
class vgg19:
	# parameters
	n_filter = [64, 128, 256, 512, 512]
	b_dropout = False
	dropout = 0.8
	#
	def __init__(self, input_x, target, n_class, is_training):
		# Currently fixed phi to phi(5, 4) [to be generalized]
		self.pred, self.phi = self.build_model(input_x, is_training)
		self.loss = self.inference_loss(self.pred, target, n_class)
	#
	def build_model(self, input_x, is_training=None):
		with tf.variable_scope('vgg19', reuse=reuse):
			phi = list()
			with tf.variable_scope('block_1'):
				# 2 conv + 1 maxpool
				conv1_1 = tf.layers.conv2d(
						  inputs=input_x,
						  filters=self.n_filter[0],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv1' )
				
				conv1_2 = tf.layers.conv2d(
						  inputs=conv1_1,
						  filters=self.n_filter[0],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv2' )
				
				phi.append(conv1_2)

				pool1   = tf.layers.max_pooling2d(
						  inputs=conv1_2, 
						  pool_size=[2, 2], 
						  strides=2, 
						  name='pool')
			#
			with tf.variable_scope('block_2'):
				# 2 conv + 1 maxpool
				conv2_1 = tf.layers.conv2d(
						  inputs=pool1,
						  filters=self.n_filter[1],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv1' )
				
				conv2_2 = tf.layers.conv2d(
						  inputs=conv2_1,
						  filters=self.n_filter[1],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv2' )
				
				phi.append(conv2_2)
				
				pool2   = tf.layers.max_pooling2d(
						  inputs=conv2_2, 
						  pool_size=[2, 2], 
						  strides=2, 
						  name='pool')
			#
			with tf.variable_scope('block_3'):
				# 4 conv + 1 maxpool
				conv3_1 = tf.layers.conv2d(
						  inputs=pool2,
						  filters=self.n_filter[2],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv1' )
				
				conv3_2 = tf.layers.conv2d(
						  inputs=conv3_1,
						  filters=self.n_filter[2],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv2' )
				
				conv3_3 = tf.layers.conv2d(
						  inputs=conv3_2,
						  filters=self.n_filter[2],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv3' )
				
				conv3_4 = tf.layers.conv2d(
						  inputs=conv3_3,
						  filters=self.n_filter[2],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv4' )
				
				phi.append(conv3_4)

				pool3   = tf.layers.max_pooling2d(
						  inputs=conv3_4, 
						  pool_size=[2, 2], 
						  strides=2, 
						  name='pool')
			#
			with tf.variable_scope('block_4'):
				# 4 conv + 1 maxpool
				conv4_1 = tf.layers.conv2d(
						  inputs=pool3,
						  filters=self.n_filter[3],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv1' )
				
				conv4_2 = tf.layers.conv2d(
						  inputs=conv4_1,
						  filters=self.n_filter[3],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv2' )
				
				conv4_3 = tf.layers.conv2d(
						  inputs=conv4_2,
						  filters=self.n_filter[3],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv3' )
				
				conv4_4 = tf.layers.conv2d(
						  inputs=conv4_3,
						  filters=self.n_filter[3],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv4' )
				
				phi.append(conv4_4)

				pool4   = tf.layers.max_pooling2d(
						  inputs=conv4_4, 
						  pool_size=[2, 2], 
						  strides=2, 
						  name='pool')
			#
			with tf.variable_scope('block_5'):
				# 4 conv + 1 maxpool
				conv5_1 = tf.layers.conv2d(
						  inputs=pool4,
						  filters=self.n_filter[4],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv1' )
				
				conv5_2 = tf.layers.conv2d(
						  inputs=conv5_1,
						  filters=self.n_filter[4],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv2' )
				
				conv5_3 = tf.layers.conv2d(
						  inputs=conv5_2,
						  filters=self.n_filter[4],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv3' )
				
				conv5_4 = tf.layers.conv2d(
						  inputs=conv5_3,
						  filters=self.n_filter[4],
						  kernel_size=[3, 3],
						  padding="same",
						  activation=tf.nn.relu,
						  name='conv4' )
				
				phi.append(conv5_4)

				pool5   = tf.layers.max_pooling2d(
						  inputs=conv5_4, 
						  pool_size=[2, 2], 
						  strides=2, 
						  name='pool' )
			#
			with tf.variable_scope('fully_connected'):
				#flatten = tf.contrib.layers.flatten(pool5)
				fc1 = tf.contrib.layers.fully_connected(
					  inputs=pool5,
					  num_outputs=4096,
					  activation_fn=tf.nn.relu )
				
				fc1 = tf.layers.batch_normalization(
					  fc1,
					  axis=-1,
					  training=is_training,
					  name='fc1_bn', 
					  renorm=False, 
					  renorm_clipping=None,
					  renorm_momentum=0.99 )
				
				if self.b_dropout:
					fc1 = tf.nn.dropout(fc1, self.dropout)
				#
				fc2 = tf.contrib.layers.fully_connected(
					  inputs=fc1,
					  num_outputs=4096,
					  activation_fn=tf.nn.relu )
				
				fc2 = tf.layers.batch_normalization(
					  fc2,
					  axis=-1,
					  training=is_training,
					  name='fc2_bn', 
					  renorm=False, 
					  renorm_clipping=None,
					  renorm_momentum=0.99 )
				
				if self.b_dropout:
					fc2 = tf.nn.dropout(fc2, self.dropout)
				#
				pred = tf.contrib.layers.fully_connected(
					   inputs=fc2,
					   num_outputs=n_class,
					   activation_fn=tf.nn.softmax )
		
		return pred, phi

	#
	def inference_loss(self, pred, target, n_class):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
						labels=tf.one_hot(target, n_class),
						logits=pred)
		return tf.reduce_mean(cross_entropy)
