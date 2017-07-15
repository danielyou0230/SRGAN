import tensorflow as tf
import tensorlayer as tl
import numpy as np

BATCH_SIZE = 64
LR_G = 0.0001 
LR_D = 0.0001 

class srgan:
	n_filter=[64, 128, 256, 512, 512]
	epsilon=0.001
	def __init__(self, input_x, t, is_training):
		if input_x is None: return
		self.out, self.phi = self.build_model(input_x, is_training)
		self.loss = self.inference_loss(self.out, t)
"""
	def BN(self,input_x,out_size,epsilon,is_training):
		fc_mean, fc_var = tf.nn.moments(
		input_x,
		axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
					# for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
		)
		scale = tf.Variable(tf.ones([out_size]))
		shift = tf.Variable(tf.zeros([out_size]))
		# apply moving average for mean and var when train on batch
		ema = tf.train.ExponentialMovingAverage(decay=0.5)
		def mean_var_with_update():
			ema_apply_op = ema.apply([fc_mean, fc_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(fc_mean), tf.identity(fc_var)
		mean, var = tf.cond(is_training,    
					mean_var_with_update,   
					lambda: (               
						ema.average(fc_mean), 
						ema.average(fc_var)
						)    
					)

		input_x = tf.nn.BN(input_x, mean, var, shift, scale, epsilon)
		return input_x
"""
	def residual_block(self, input_x, index, out_size, is_training=True):
		with tf.variable_scope("block_" + str(index)) as scope:
			conv1 = tf.layers.conv2d(
					inputs=input_x,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv1' )

			bn   = tf.layers.batch_normalization(conv1, training=is_training)

			act_prelu = tf.contrib.keras.layers.PReLU()
			act_prelu.input(bn)
			
			conv2 = tf.layers.conv2d(
					inputs=act_prelu,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv2' )

			bn2 = tf.layers.batch_normalization(conv2, training=is_training)

			tf.add(input_x, bn2, name="elementwise_sum")
		
		return bn2


	def build(self, input_x,input_d, is_training, reuse=False):
		with tf.variable_scope('generator', reuse=reuse):
			with tf.variable_scope('ResBlock_0'):
				conv_b0_1 = tf.layers.conv2d(
					inputs=input_x,
					filters=self.n_filter[0],
					kernel_size=[9 , 9],
					padding="same",
					name='conv1' )

				act_prelu_b0 = tf.contrib.keras.layers.PReLU()
				act_prelu_b0.input(conv1)

			with tf.variable_scope("ResBlock_1"):
				conv_b1_1 = tf.layers.conv2d(
							inputs=act_prelu_b0,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b1_1' )

				bn1_1 = tf.layers.batch_normalization(conv_b1_1, training=is_training)
				
				act_prelu_b1 = tf.contrib.keras.layers.PReLU()
				act_prelu_b1.input(bn1_1)
				
				conv_b1_2 = tf.layers.conv2d(
						  inputs=act_prelu_b1,
						  filters=self.n_filter[0],
						  kernel_size=[3, 3],
						  padding="same",
						  name='conv_b1_2' )
				
				bn1_2 = tf.layers.batch_normalization(conv_b1_2, training=is_training)
				ressum_1 = tf.add(act_prelu_b0, bn1_2, name="ResSum")

			with tf.variable_scope("ResBlock_2"):
				conv_b2_1 = tf.layers.conv2d(
					inputs=ressum_1,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv_b2_1' )
				
				bn2_1 = tf.layers.batch_normalization(conv_b2_1, training=is_training)
				
				act_prelu_b2 = tf.contrib.keras.layers.PReLU()
				act_prelu_b2.input(bn2_1)
				
				conv_b2_2 = tf.layers.conv2d(
					inputs=act_prelu_b2,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv_b2_2' )
				
				bn2_2 = tf.layers.batch_normalization(conv_b2_2, training=is_training)
				ressum_2 = tf.add(ressum_1, bn2_2, name="ResSum")

			with tf.variable_scope("ResBlock_3"):
				conv_b3_1 = tf.layers.conv2d(
					inputs=ressum_2,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv_b3_1' )
				
				bn3_1 = tf.layers.batch_normalization(conv_b3_1, training=is_training)
				
				act_prelu_b3 = tf.contrib.keras.layers.PReLU()
				act_prelu_b3.input(bn3_1)

				conv_b3_2 = tf.layers.conv2d(
					inputs=act_prelu_b3,
					filters=self.n_filter[0],
					kernel_size=[3, 3],
					padding="same",
					name='conv_b3_2' )
				
				bn3_2 = tf.layers.batch_normalization(conv_b3_2, training=is_training)
				ressum_3 = tf.add(ressum_2, bn3_2, name="ResSum")

			with tf.variable_scope("ResBlock_4"):
				conv_b4_1 = tf.layers.conv2d(
							inputs=ressum_3,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b4_1' )
				
				bn4_1 = tf.layers.batch_normalization(conv_b4_1, training=is_training)
				
				act_prelu_b4 = tf.contrib.keras.layers.PReLU()
				act_prelu_b4.input(bn4_1)
				
				conv_b4_2 = tf.layers.conv2d(
							inputs=act_prelu_b4,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b4_2' )
				
				bn4_2 = tf.layers.batch_normalization(conv_b4_2, training=is_training)
				ressum_4 = tf.add(ressum_3, bn4_2, name="ResSum")

			with tf.variable_scope("ResBlock_5"):
				conv_b5_1 = tf.layers.conv2d(
							inputs=ressum_4,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b5_1' )
				
				bn5_1 = tf.layers.batch_normalization(conv_b5_1, training=is_training)
				
				act_prelu_b5 = tf.contrib.keras.layers.PReLU()
				act_prelu_b5.input(bn5_1)
				
				conv_b5_2 = tf.layers.conv2d(
							inputs=act_prelu_b5,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b5_2' )
				
				bn5_2 = tf.layers.batch_normalization(conv_b5_2, training=is_training)
				ressum_5 = tf.add(ressum_4, bn5_2, name="ResSum")

			with tf.variable_scope("ResBlock_6"):
				conv_b6_1 = tf.layers.conv2d(
							inputs=ressum_5,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b6_1' )
				
				bn6_1 = tf.layers.batch_normalization(conv_b6_1, training=is_training)
				
				act_prelu_b6 = tf.contrib.keras.layers.PReLU()
				act_prelu_b6.input(bn6_1)
				
				conv_b6_2 = tf.layers.conv2d(
							inputs=act_prelu_b6,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b6_2' )
				
				bn6_2 = tf.layers.batch_normalization(conv_b6_2, training=is_training)
				ressum_6 = tf.add(ressum_5, bn6_2, name="ResSum")

			with tf.variable_scope("ResBlock_7"):
				conv_b7_1 = tf.layers.conv2d(
							inputs=ressum_6,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b7_1' )
				
				bn7_1 = tf.layers.batch_normalization(conv_b7_1, training=is_training)
				
				act_prelu_b7 = tf.contrib.keras.layers.PReLU()
				act_prelu_b7.input(bn7_1)
				
				conv_b7_2 = tf.layers.conv2d(
							inputs=act_prelu_b7,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b7_2' )
				
				bn7_2 = tf.layers.batch_normalization(conv_b7_2, training=is_training)
				ressum_7 = tf.add(ressum_6, bn7_2, name="ResSum")

			with tf.variable_scope("ResBlock_8"):
				conv_b8_1 = tf.layers.conv2d(
							inputs=ressum_7,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b8_1' )
				
				bn8_1 = tf.layers.batch_normalization(conv_b8_1, training=is_training)
				
				act_prelu_b8 = tf.contrib.keras.layers.PReLU()
				act_prelu_b8.input(bn8_1)
				
				conv_b8_2 = tf.layers.conv2d(
							inputs=act_prelu_b8,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b8_2' )
				
				bn8_2 = tf.layers.batch_normalization(conv_b8_2, training=is_training)
				ressum_8 = tf.add(ressum_7, bn8_2, name="ResSum")

			with tf.variable_scope("ResBlock_9"):
				conv_b9_1 = tf.layers.conv2d(
							inputs=ressum_8,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b9_1' )
				
				bn9_1 = tf.layers.batch_normalization(conv_b9_1, training=is_training)
				
				act_prelu_b9 = tf.contrib.keras.layers.PReLU()
				act_prelu_b9.input(bn9_1)
				
				conv_b9_2 = tf.layers.conv2d(
							inputs=act_prelu_b9,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_b9_2' )
				
				bn9_2 = tf.layers.batch_normalization(conv_b9_2, training=is_training)
				ressum_9 = tf.add(ressum_8, bn9_2, name="ResSum")

			with tf.variable_scope("ResBlock_10"):
				conv_b10_1 = tf.layers.conv2d(
							 inputs=ressum_9,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b10_1' )
				
				bn10_1 = tf.layers.batch_normalization(conv_b10_1, training=is_training)
				
				act_prelu_b10 = tf.contrib.keras.layers.PReLU()
				act_prelu_b10.input(bn10_1)
				
				conv_b10_2 = tf.layers.conv2d(
							 inputs=act_prelu_b10,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b10_2' )
				
				bn10_2 = tf.layers.batch_normalization(conv_b10_2, training=is_training)
				ressum_10 = tf.add(ressum_9, bn10_2, name="ResSum")

			with tf.variable_scope("ResBlock_11"):
				conv_b11_1 = tf.layers.conv2d(
							 inputs=ressum_10,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b11_1' )
				
				bn11_1 = tf.layers.batch_normalization(conv_b11_1, training=is_training)
				
				act_prelu_b11 = tf.contrib.keras.layers.PReLU()
				act_prelu_b11.input(bn11_1)
				
				conv_b11_2 = tf.layers.conv2d(
							 inputs=act_prelu_b11,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b11_2' )
				
				bn11_2 = tf.layers.batch_normalization(conv_b11_2, training=is_training)
				ressum_11 = tf.add(ressum_10, bn11_2, name="ResSum")

			with tf.variable_scope("ResBlock_12"):
				conv_b12_1 = tf.layers.conv2d(
							 inputs=ressum_11,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b12_1' )
				bn12_1 = tf.layers.batch_normalization(conv_b12_1, training=is_training)
				
				act_prelu_b12 = tf.contrib.keras.layers.PReLU()
				act_prelu_b12.input(bn12_1)
				
				conv_b12_2 = tf.layers.conv2d(
							 inputs=act_prelu_b12,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b12_2' )
				
				bn12_2 = tf.layers.batch_normalization(conv_b12_2, training=is_training)
				ressum_12 = tf.add(ressum_11, bn12_2, name="ResSum")

			with tf.variable_scope("ResBlock_13"):
				conv_b13_1 = tf.layers.conv2d(
							 inputs=ressum_12,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b13_1' )
				
				bn13_1 = tf.layers.batch_normalization(conv_b13_1, training=is_training)
				
				act_prelu_b13 = tf.contrib.keras.layers.PReLU()
				act_prelu_b13.input(bn13_1)
				
				conv_b13_2 = tf.layers.conv2d(
							 inputs=act_prelu_b13,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b13_2' )
				
				bn13_2 = tf.layers.batch_normalization(conv_b13_2, training=is_training)
				ressum_13 = tf.add(ressum_12, bn13_2, name="ResSum")

			with tf.variable_scope("ResBlock_14"):
				conv_b14_1 = tf.layers.conv2d(
							 inputs=ressum_13,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b14_1' )
				bn14_1 = tf.layers.batch_normalization(conv_b14_1, training=is_training)
				act_prelu_b14 = tf.contrib.keras.layers.PReLU()
				act_prelu_b14.input(bn14_1)
				conv_b14_2 = tf.layers.conv2d(
							 inputs=act_prelu_b14,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b14_2' )
				bn14_2 = tf.layers.batch_normalization(conv_b14_2, training=is_training)
				ressum_14 = tf.add(ressum_13, bn14_2, name="ResSum")

			with tf.variable_scope("ResBlock_15"):
				conv_b15_1 = tf.layers.conv2d(
							 inputs=ressum_14,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b15_1' )
				bn15_1 = tf.layers.batch_normalization(conv_b15_1, training=is_training)
				act_prelu_b15 = tf.contrib.keras.layers.PReLU()
				act_prelu_b15.input(bn15_1)
				conv_b15_2 = tf.layers.conv2d(
							 inputs=act_prelu_b15,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b15_2' )
				bn15_2 = tf.layers.batch_normalization(conv_b15_2, training=is_training)
				ressum_15 = tf.add(ressum_14, bn15_2, name="ResSum")

			with tf.variable_scope("ResBlock_16"):
				conv_b16_1 = tf.layers.conv2d(
							 inputs=ressum_15,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b16_1' )
				
				bn16_1 = tf.layers.batch_normalization(conv_b16_1, training=is_training)
				
				act_prelu_b16 = tf.contrib.keras.layers.PReLU()
				act_prelu_b16.input(bn16_1)
				
				conv_b16_2 = tf.layers.conv2d(
							 inputs=act_prelu_b16,
							 filters=self.n_filter[0],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_b16_2' )
				
				bn16_2 = tf.layers.batch_normalization(conv_b16_2, training=is_training)
				ressum_16 = tf.add(ressum_15, bn16_2, name="ResSum")
			#
			with tf.variable_scope("gblock1"):
				conv_gb_1 = tf.layers.conv2d(
							inputs=ressum_16,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_gb1' )
				
				bn_gb1 = tf.layers.batch_normalization(conv_gb1_1, training=is_training)
				ressum_gb1 = tf.add(act_prelu_b0, bn_gb1, name="ResSum")
			#
			with tf.variable_scope('gblock2'):
				conv_gb_2 = tf.layers.conv2d(
							 inputs=ressum_gb1,
							 filters=self.n_filter[2],
							 kernel_size=[3, 3],
							 padding="same",
							 name='conv_gb2' )
				
				px_shuffler_2 = SubpixelConv2d(
								conv_gb_2, 
								scale=2, 
								n_out_channel=None, 
								act=tf.nn.relu, 
								name='pixelshufflerx2/1')

				gb2_act = tf.contrib.keras.layers.PReLU()
				gb2_act.input(px_shuffler_2)
			#
			with tf.variable_scope('gblock3'):
				conv_gb_3 = tf.layers.conv2d(
							inputs=gb2_act,
							filters=self.n_filter[2],
							kernel_size=[3, 3],
							padding="same",
							name='conv_gb3' )
				
				px_shuffler_3 = SubpixelConv2d(
								conv_gb_3, 
								scale=2, 
								n_out_channel=None, 
								act=tf.nn.relu, 
								name='pixelshufflerx2/2')
				
				gb3_act = tf.contrib.keras.layers.PReLU()
				gb3_act.input(px_shuffler_3)
			#
			with tf.variable_scope('gblock4'):
				conv_gb_4 = tf.layers.conv2d(
							inputs=gb3_act,
							filters=3,
							kernel_size=[9, 9],
							padding="same",
							name='conv_gb3' )




		with tf.variable_scope('discriminator', reuse=reuse):
			with tf.variable_scope('block0'):
				conv0 = tf.layers.conv2d(
							inputs=input_d,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							padding="same",
							name='conv_0' )
				b0_act = tf.contrib.keras.layers.LeakyReLU()
				b0_act.input(conv_0)

			with tf.variable_scope('block1'):
				conv1 = tf.layers.conv2d(
							inputs=b0_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv1' )
				bn1 = tf.layers.batch_normalization(conv1, training=is_training)
				b1_act = tf.contrib.keras.layers.LeakyReLU()
				b1_act.input(bn1)

			with tf.variable_scope('block2'):
				conv2 = tf.layers.conv2d(
							inputs=b1_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv2' )
				bn2 = tf.layers.batch_normalization(conv2, training=is_training)
				b2_act = tf.contrib.keras.layers.LeakyReLU()
				b2_act.input(bn2)

			with tf.variable_scope('block3'):
				conv3 = tf.layers.conv2d(
							inputs=b2_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv3' )
				bn3 = tf.layers.batch_normalization(conv3, training=is_training)
				b3_act = tf.contrib.keras.layers.LeakyReLU()
				b3_act.input(bn3)

			with tf.variable_scope('block4'):
				conv4 = tf.layers.conv2d(
							inputs=b3_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv4' )
				bn4 = tf.layers.batch_normalization(conv4, training=is_training)
				b4_act = tf.contrib.keras.layers.LeakyReLU()
				b4_act.input(bn4)

			with tf.variable_scope('block5'):
				conv5 = tf.layers.conv2d(
							inputs=b4_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv5' )
				bn5 = tf.layers.batch_normalization(conv5, training=is_training)
				b5_act = tf.contrib.keras.layers.LeakyReLU()
				b5_act.input(bn5)

			with tf.variable_scope('block6'):
				conv6 = tf.layers.conv2d(
							inputs=b5_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv6' )
				bn6 = tf.layers.batch_normalization(conv6, training=is_training)
				b6_act = tf.contrib.keras.layers.LeakyReLU()
				b6_act.input(bn6)

			with tf.variable_scope('block7'):
				conv7 = tf.layers.conv2d(
							inputs=b6_act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="same",
							name='conv7' )
				bn7 = tf.layers.batch_normalization(conv7, training=is_training)
				b7_act = tf.contrib.keras.layers.LeakyReLU()
				b7_act.input(bn7)

			with tf.variable_scope('block8'):
				#Dense(1024) LeakyRelu Dense(1) sigmoid




	def inference_losses(self, inputs, imitation, true_output, fake_output):
		def inference_content_loss(input_x, imitation):
			_, inputs_phi = self.vgg.build_model(
				inputs, tf.constant(False), False) # First
			_, imitation_phi = self.vgg.build_model(
				imitation, tf.constant(False), True) # Second

			content_loss = None
			for i in range(len(x_phi)):
				l2_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
				if content_loss is None:
					content_loss = l2_loss
				else:
					content_loss = content_loss + l2_loss
			return tf.reduce_mean(content_loss)

		def inference_adversarial_loss(real_output, fake_output):
			alpha = 1e-5
			g_loss = tf.reduce_mean(
				tf.nn.l2_loss(fake_output - tf.ones_like(fake_output)))
			d_loss_real = tf.reduce_mean(
				tf.nn.l2_loss(real_output - tf.ones_like(true_output)))
			d_loss_fake = tf.reduce_mean(
				tf.nn.l2_loss(fake_output + tf.zeros_like(fake_output)))
			d_loss = d_loss_real + d_loss_fake
			return (g_loss * alpha, d_loss * alpha)

		def inference_adversarial_loss_with_sigmoid(real_output, fake_output):
			alpha = 1e-3
			g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
				labels=tf.ones_like(fake_output),
				logits=fake_output)
			d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
				labels=tf.ones_like(real_output),
				logits=real_output)
			d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
				labels=tf.zeros_like(fake_output),
				logits=fake_output)
			d_loss = d_loss_real + d_loss_fake
			return (g_loss * alpha, d_loss * alpha)

		content_loss = inference_content_loss(input_x, imitation)
		generator_loss, discriminator_loss = (
			inference_adversarial_loss(true_output, fake_output))
		g_loss = content_loss + generator_loss
		d_loss = discriminator_loss
		return (g_loss, d_loss)
