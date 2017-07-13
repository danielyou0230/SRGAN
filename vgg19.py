import tensorflow as tf

# VGG19
class vgg19:
	# parameters
	n_filter = [64, 128, 256, 512, 512]
	dropout = 0.8
	#
	def __init__(self, arg):
		super(vgg19, self).__init__()
		self.arg = arg
		
	def build(self, input_x, is_training=None):
		with name_scope('block_1') as scope:
			# 2 conv + 1 maxpool
			conv1_1 = tf.layers.conv2d(
					  inputs=input_x,
					  filters=self.n_filter[0],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv1_1' )

			conv1_2 = tf.layers.conv2d(
					  inputs=conv1_1,
					  filters=self.n_filter[0],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv1_2' )
		
			pool1   = tf.layers.max_pooling2d(
					  inputs=conv1_2, 
					  pool_size=[2, 2], 
					  strides=2, 
					  name='pool1')
		
		with name_scope('block_2') as scope:
			# 2 conv + 1 maxpool
			conv2_1 = tf.layers.conv2d(
					  inputs=pool1,
					  filters=self.n_filter[1],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv2_1' )

			conv2_2 = tf.layers.conv2d(
					  inputs=conv2_1,
					  filters=self.n_filter[1],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv2_2' )
			
			pool2   = tf.layers.max_pooling2d(
					  inputs=conv2_2, 
					  pool_size=[2, 2], 
					  strides=2, 
					  name='pool2')
		
		with name_scope('block_3') as scope:
			# 4 conv + 1 maxpool
			conv3_1 = tf.layers.conv2d(
					  inputs=pool2,
					  filters=self.n_filter[2],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv3_1' )
			
			conv3_2 = tf.layers.conv2d(
					  inputs=conv3_1,
					  filters=self.n_filter[2],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv3_2' )
			
			conv3_3 = tf.layers.conv2d(
					  inputs=conv3_2,
					  filters=self.n_filter[2],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv3_3' )
			
			conv3_4 = tf.layers.conv2d(
					  inputs=conv3_3,
					  filters=self.n_filter[2],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv3_4' )
			
			pool3   = tf.layers.max_pooling2d(
					  inputs=conv3_4, 
					  pool_size=[2, 2], 
					  strides=2, 
					  name='pool3')

		with name_scope('block_4') as scope:
			# 4 conv + 1 maxpool
			conv4_1 = tf.layers.conv2d(
					  inputs=pool3,
					  filters=self.n_filter[3],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv4_1' )
			
			conv4_2 = tf.layers.conv2d(
					  inputs=conv4_1,
					  filters=self.n_filter[3],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv4_2' )
			
			conv4_3 = tf.layers.conv2d(
					  inputs=conv4_2,
					  filters=self.n_filter[3],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv4_3' )
			
			conv4_4 = tf.layers.conv2d(
					  inputs=conv4_3,
					  filters=self.n_filter[3],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv4_4' )
			
			pool4   = tf.layers.max_pooling2d(
					  inputs=conv4_4, 
					  pool_size=[2, 2], 
					  strides=2, 
					  name='pool4')

		with name_scope('block_5') as scope:
			# 4 conv + 1 maxpool
			conv5_1 = tf.layers.conv2d(
					  inputs=pool4,
					  filters=self.n_filter[4],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv5_1' )
			
			conv5_2 = tf.layers.conv2d(
					  inputs=conv5_1,
					  filters=self.n_filter[4],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv5_2' )
			
			conv5_3 = tf.layers.conv2d(
					  inputs=conv5_2,
					  filters=self.n_filter[4],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv5_3' )
			
			conv5_4 = tf.layers.conv2d(
					  inputs=conv5_3,
					  filters=self.n_filter[4],
					  kernel_size=[3, 3],
					  padding="same",
					  activation=tf.nn.relu,
					  name='conv5_4' )
			
			pool5   = tf.layers.max_pooling2d(
					  inputs=conv5_4, 
					  pool_size=[2, 2], 
					  strides=2, 
					  name='pool5' )

		with name_scope('fully_connected') as scope:
			#flatten = tf.contrib.layers.flatten(pool5)
			fc1 = tf.contrib.layers.fully_connected(
				  inputs=pool5,
				  num_outputs=4096,
				  activation_fn=tf.nn.relu )
			fc1 = tf.layers.batch_normalization(
				  fc1,
				  axis=-1,
				  momentum=0.99,
				  epsilon=0.001,
				  training=is_training,
				  name='fc1_bn',
				  renorm=False,
				  renorm_clipping=None,
				  renorm_momentum=0.99)
			#fc1 = tf.nn.dropout(fc1, self.dropout)
			#
			fc2 = tf.contrib.layers.fully_connected(
				  inputs=pool5,
				  num_outputs=4096,
				  activation_fn=tf.nn.relu )
			fc2 = tf.layers.batch_normalization()
			#fc2 = tf.nn.dropout(fc2, self.dropout)
			#
			fc3 = tf.contrib.layers.fully_connected(
				   inputs=pool5,
				   num_outputs=1000,
				   activation_fn=tf.nn.softmax )
			fc3 = tf.layers.batch_normalization()

