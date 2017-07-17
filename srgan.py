import tensorflow as tf
import numpy as np
from tensorlayer.layers import SubpixelConv2d
from PIL import Image
from vgg19 import vgg19

BATCH_SIZE = 64
LR_G = 0.0001 
LR_D = 0.0001 

class srgan:
	n_filter = [64, 128, 256, 512, 512]
	epsilon  = 0.001
	sr_rate = 4
	def __init__(self, input_x, is_training):
		self.vgg = vgg19(None, None, None)
		self.downscaled = self.downscale(input_x)
		self.imitation = self.generator(self.downscaled, is_training, False)
		self.real_output = self.discriminator(input_x, is_training, False)
		self.fake_output = self.discriminator(self.imitation, is_training, True)
		self.g_loss, self.d_loss = self.inference_losses(
			input_x, self.imitation, self.real_output, self.fake_output)
"""
	def BN(self,input_x,out_size,epsilon,is_training):
		fc_mean, fc_var = tf.nn.moments(
		input_x,
		axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
		# for image, you wanna do [0, 1, 2] 
		# for [batch, height, width] but not channel
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
	def residual_block(self, input_x, index, is_training=True):
		with tf.variable_scope("ResBlock_" + str(index)) as scope:
			conv1 = tf.layers.conv2d(
						inputs=input_x,
						filters=self.n_filter[index],
						kernel_size=[3, 3],
						padding="SAME",
						name='conv1' )

			conv1 = tf.layers.batch_normalization(conv1, training=is_training)
			act_prelu = tf.contrib.keras.layers.PReLU()
			act_prelu.input(conv1)

			conv2 = tf.layers.conv2d(
						inputs=act_prelu,
						filters=self.n_filter[index],
						kernel_size=[3, 3],
						padding="SAME",
						name='conv2' )
			conv2 = tf.layers.batch_normalization(conv2, training=is_training)
			res   = tf.add(input_x, conv2, name="elementwise_sum")
		return res

	def generator(self, input_x, is_training, reuse=False):
		# Generator
		with tf.variable_scope('generator', reuse=reuse):
			with tf.variable_scope('iniBlock'):
				conv0 = tf.layers.conv2d(
								inputs=input_x,
								filters=self.n_filter[0],
								kernel_size=[9 , 9],
								padding="SAME",
								name='conv01' )

				act_0 = tf.contrib.keras.layers.PReLU()
				act_0.input(conv0)

			with tf.variable_scope("ResBlock_01"):
				conv01_1 = tf.layers.conv2d(
								inputs=act_0,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv01_1' )

				conv01_1 = tf.layers.batch_normalization(
								conv01_1, 
								axis=-1,
								training=is_training )
				
				act_01 = tf.contrib.keras.layers.PReLU()
				act_01.input(conv01_1)
				
				conv01_2 = tf.layers.conv2d(
								inputs=act_01,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv01_2' )
				
				conv01_2 = tf.layers.batch_normalization(
								conv01_2, 
								axis=-1,
								training=is_training )
				ressum01 = tf.add(act_00, conv01_2, name="ResSum01")

			with tf.variable_scope("ResBlock_02"):
				conv02_1 = tf.layers.conv2d(
								inputs=ressum01,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv02_1' )
				
				conv02_1 = tf.layers.batch_normalization(
								conv02_1, 
								axis=-1,
								training=is_training )
				
				act_02 = tf.contrib.keras.layers.PReLU()
				act_02.input(conv02_1)
				
				conv02_2 = tf.layers.conv2d(
								sinputs=act_02,
								sfilters=self.n_filter[0],
								skernel_size=[3, 3],
								spadding="SAME",
								sname='conv02_2' )
				
				conv02_2 = tf.layers.batch_normalization(
								conv02_2, 
								axis=-1,
								training=is_training )
				ressum02 = tf.add(ressum01, conv02_2, name="ResSum02")

			with tf.variable_scope("ResBlock_03"):
				conv03_1 = tf.layers.conv2d(
								inputs=ressum02,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv03_1' )
				
				conv03_1 = tf.layers.batch_normalization(
								conv03_1, 
								axis=-1,
								training=is_training )
				
				act_03 = tf.contrib.keras.layers.PReLU()
				act_03.input(conv03_1)

				conv03_2 = tf.layers.conv2d(
								inputs=act_03,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv03_2' )
				
				conv03_2 = tf.layers.batch_normalization(
								conv03_2, 
								axis=-1,
								training=is_training )
				ressum03 = tf.add(ressum02, conv03_2, name="ResSum03")

			with tf.variable_scope("ResBlock_04"):
				conv04_1 = tf.layers.conv2d(
								inputs=ressum03,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv04_1' )
				
				conv04_1 = tf.layers.batch_normalization(
								conv04_1, 
								axis=-1,
								training=is_training )
				
				act_04 = tf.contrib.keras.layers.PReLU()
				act_04.input(conv04_1)
				
				conv04_2 = tf.layers.conv2d(
								inputs=act_04,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv04_2' )
				
				conv04_2 = tf.layers.batch_normalization(
								conv04_2, 
								axis=-1,
								training=is_training )
				ressum04 = tf.add(ressum03, conv04_2, name="ResSum04")

			with tf.variable_scope("ResBlock_05"):
				conv05_1 = tf.layers.conv2d(
								inputs=ressum04,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv05_1' )
				
				conv05_1 = tf.layers.batch_normalization(
								conv05_1, training=is_training)
				
				act_05 = tf.contrib.keras.layers.PReLU()
				act_05.input(conv05_1)
				
				conv05_2 = tf.layers.conv2d(
								inputs=act_05,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv05_2' )
				
				conv05_2 = tf.layers.batch_normalization(
								conv05_2, 
								axis=-1,
								training=is_training )
				ressum05 = tf.add(ressum04, conv05_2, name="ResSum05")

			with tf.variable_scope("ResBlock_06"):
				conv06_1 = tf.layers.conv2d(
								inputs=ressum05,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv06_1' )
				
				conv06_1 = tf.layers.batch_normalization(
								conv06_1, 
								axis=-1,
								training=is_training )
				
				act_06 = tf.contrib.keras.layers.PReLU()
				act_06.input(conv06_1)
				
				conv06_2 = tf.layers.conv2d(
								inputs=act_06,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv06_2' )
				
				conv06_2 = tf.layers.batch_normalization(
								conv06_2, 
								axis=-1,
								training=is_training )
				ressum06 = tf.add(ressum05, conv06_2, name="ResSum06")

			with tf.variable_scope("ResBlock_07"):
				conv07_1 = tf.layers.conv2d(
								inputs=ressum06,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv07_1' )
				
				conv07_1 = tf.layers.batch_normalization(
								conv07_1, 
								axis=-1,
								training=is_training )
				
				act_07 = tf.contrib.keras.layers.PReLU()
				act_07.input(conv07_1)
				
				conv07_2 = tf.layers.conv2d(
								inputs=act_07,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv07_2' )
				
				conv07_2 = tf.layers.batch_normalization(
								conv07_2, 
								axis=-1,
								training=is_training )
				ressum07 = tf.add(ressum06, conv07_2, name="ResSum07")

			with tf.variable_scope("ResBlock_08"):
				conv08_1 = tf.layers.conv2d(
								inputs=ressum07,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv08_1' )
				
				conv08_1 = tf.layers.batch_normalization(
								conv08_1, 
								axis=-1,
								training=is_training )
				
				act_08 = tf.contrib.keras.layers.PReLU()
				act_08.input(conv08_1)
				
				conv08_2 = tf.layers.conv2d(
								inputs=act_08,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv08_2' )
				
				conv08_2 = tf.layers.batch_normalization(
								conv08_2, 
								axis=-1,
								training=is_training )
				ressum08 = tf.add(ressum07, conv08_2, name="ResSum08")

			with tf.variable_scope("ResBlock_09"):
				conv09_1 = tf.layers.conv2d(
								inputs=ressum08,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv09_1' )
				
				conv09_1 = tf.layers.batch_normalization(
								conv09_1, 
								axis=-1,
								training=is_training )
				
				act_09 = tf.contrib.keras.layers.PReLU()
				act_09.input(conv09_1)
				
				conv09_2 = tf.layers.conv2d(
								inputs=act_09,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv09_2' )
				
				conv09_2 = tf.layers.batch_normalization(
								conv09_2, 
								axis=-1,
								training=is_training )
				ressum09 = tf.add(ressum08, conv09_2, name="ResSum09")

			with tf.variable_scope("ResBlock_10"):
				conv10_1 = tf.layers.conv2d(
								inputs=ressum09,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv10_1' )
				
				conv10_1 = tf.layers.batch_normalization(
								conv10_1, 
								axis=-1,
								training=is_training )
				
				act_10 = tf.contrib.keras.layers.PReLU()
				act_10.input(conv10_1)
				
				conv10_2 = tf.layers.conv2d(
								inputs=act_10,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv10_2' )
				
				conv10_2 = tf.layers.batch_normalization(
								conv10_2, 
								axis=-1,
								training=is_training )
				ressum10 = tf.add(ressum09, conv10_2, name="ResSum10")

			with tf.variable_scope("ResBlock_11"):
				conv11_1 = tf.layers.conv2d(
								inputs=ressum10,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv11_1' )
				
				conv11_1 = tf.layers.batch_normalization(
								conv11_1, 
								axis=-1,
								training=is_training )
				
				act_11 = tf.contrib.keras.layers.PReLU()
				act_11.input(conv11_1)
				
				conv11_2 = tf.layers.conv2d(
								inputs=act_11,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv11_2' )
				
				conv11_2 = tf.layers.batch_normalization(
								conv11_2, 
								axis=-1,
								training=is_training )
				ressum11 = tf.add(ressum10, conv11_2, name="ResSum11")

			with tf.variable_scope("ResBlock_12"):
				conv12_1 = tf.layers.conv2d(
								inputs=ressum11,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv12_1' )
				conv12_1 = tf.layers.batch_normalization(
								conv12_1, 
								axis=-1,
								training=is_training )

				act_12 = tf.contrib.keras.layers.PReLU()
				act_12.input(conv12_1)
				
				conv12_2 = tf.layers.conv2d(
								inputs=act_12,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv12_2' )
				
				conv12_2 = tf.layers.batch_normalization(
								conv12_2, 
								axis=-1,
								training=is_training )
				ressum12 = tf.add(ressum11, conv12_2, name="ResSum12")

			with tf.variable_scope("ResBlock_13"):
				conv13_1 = tf.layers.conv2d(
								inputs=ressum12,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv13_1' )
				
				conv13_1 = tf.layers.batch_normalization(
								conv13_1, 
								axis=-1,
								training=is_training )
				
				act_13 = tf.contrib.keras.layers.PReLU()
				act_13.input(conv13_1)
				
				conv13_2 = tf.layers.conv2d(
								inputs=act_13,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv13_2' )
				
				conv13_2 = tf.layers.batch_normalization(
								conv13_2, 
								axis=-1,
								training=is_training )
				ressum13 = tf.add(ressum12, conv13_2, name="ResSum13")

			with tf.variable_scope("ResBlock_14"):
				conv14_1 = tf.layers.conv2d(
								inputs=ressum13,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv14_1' )
				conv14_1 = tf.layers.batch_normalization(
								conv14_1, 
								axis=-1,
								training=is_training )
				act_14 = tf.contrib.keras.layers.PReLU()
				act_14.input(conv14_1)

				conv14_2 = tf.layers.conv2d(
								inputs=act_14,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv14_2' )
				conv14_2 = tf.layers.batch_normalization(
								conv14_2, 
								axis=-1,
								training=is_training )
				ressum14 = tf.add(ressum13, conv14_2, name="ResSum14")

			with tf.variable_scope("ResBlock_15"):
				conv15_1 = tf.layers.conv2d(
								inputs=ressum14,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv15_1' )
				conv15_1 = tf.layers.batch_normalization(
								conv15_1, 
								axis=-1,
								training=is_training )
				act_15 = tf.contrib.keras.layers.PReLU()
				act_15.input(conv15_1)

				conv15_2 = tf.layers.conv2d(
								inputs=act_15,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv15_2' )
				conv15_2 = tf.layers.batch_normalization(
								conv15_2, 
								axis=-1,
								training=is_training )
				ressum15 = tf.add(ressum14, conv15_2, name="ResSum15")

			with tf.variable_scope("ResBlock_16"):
				conv16_1 = tf.layers.conv2d(
								inputs=ressum15,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv16_1' )
				
				conv16_1 = tf.layers.batch_normalization(
								conv16_1, 
								axis=-1,
								training=is_training )
				
				act_16 = tf.contrib.keras.layers.PReLU()
				act_16.input(conv16_1)
				
				conv16_2 = tf.layers.conv2d(
								inputs=act_16,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv16_2' )
				
				conv16_2 = tf.layers.batch_normalization(
								conv16_2, 
								axis=-1,
								training=is_training )
				ressum16 = tf.add(ressum15, conv16_2, name="ResSum16")
			#
			with tf.variable_scope("gBlock1"):
				conv_gB1 = tf.layers.conv2d(
								inputs=ressum16,
								filters=self.n_filter[0],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv_gB1' )
				
				conv_gB1 = tf.layers.batch_normalization(
								conv_gB1, 
								axis=-1,
								training=is_training )
				ressum_gB1 = tf.add(act_0, conv_gB1, name="ResSum")
			#
			with tf.variable_scope('gBlock2'):
				conv_gB2 = tf.layers.conv2d(
								inputs=ressum_gB1,
								filters=self.n_filter[2],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv_gB2' )
				
				conv_gB2 = SubpixelConv2d(
								conv_gB2, 
								scale=2, 
								n_out_channel=None, 
								act=tf.nn.relu, 
								name='pixelshufflerx2/1')

				act_gB2 = tf.contrib.keras.layers.PReLU()
				act_gB2.input(conv_gB2)
			#
			with tf.variable_scope('gBlock3'):
				conv_gB3 = tf.layers.conv2d(
								inputs=act_gB2,
								filters=self.n_filter[2],
								kernel_size=[3, 3],
								padding="SAME",
								name='conv_gB3' )
				
				conv_gB3 = SubpixelConv2d(
								conv_gB3, 
								scale=2, 
								n_out_channel=None, 
								act=tf.nn.relu, 
								name='PixelShufflerX2/2' )
				
				act_gB3 = tf.contrib.keras.layers.PReLU()
				act_gB3.input(conv_gB3)
			#
			with tf.variable_scope('conv_final'):
				conv = tf.layers.conv2d(
								inputs=act_gB3,
								filters=3,
								kernel_size=[9, 9],
								padding="SAME",
								name='conv' )

		return conv

	def discriminator(self, input_x, is_training, reuse=False)
		# Discriminator
		with tf.variable_scope('discriminator', reuse=reuse):
			with tf.variable_scope('Block_0'):
				conv0 = tf.layers.conv2d(
							inputs=input_x,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(1, 1),
							padding="SAME",
							name='conv0' )
				b0act = tf.contrib.keras.layers.LeakyReLU()
				b0act.input(conv0)

			with tf.variable_scope('Block_1'):
				conv1 = tf.layers.conv2d(
							inputs=b0act,
							filters=self.n_filter[0],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="SAME",
							name='conv1' )
				conv1 = tf.layers.batch_normalization(
							conv1, 
							axis=-1,
							training=is_training )
				b1act = tf.contrib.keras.layers.LeakyReLU()
				b1act.input(conv1)

			with tf.variable_scope('Block_2'):
				conv2 = tf.layers.conv2d(
							inputs=b1act,
							filters=self.n_filter[1],
							kernel_size=[3, 3],
							strides=(1, 1),
							padding="SAME",
							name='conv2' )
				conv2 = tf.layers.batch_normalization(
							conv2, 
							axis=-1,
							training=is_training )
				b2act = tf.contrib.keras.layers.LeakyReLU()
				b2act.input(conv2)

			with tf.variable_scope('Block_3'):
				conv3 = tf.layers.conv2d(
							inputs=b2act,
							filters=self.n_filter[1],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="SAME",
							name='conv3' )
				conv3 = tf.layers.batch_normalization(
							conv3, 
							axis=-1,
							training=is_training )
				b3act = tf.contrib.keras.layers.LeakyReLU()
				b3act.input(conv3)

			with tf.variable_scope('Block_4'):
				conv4 = tf.layers.conv2d(
							inputs=b3act,
							filters=self.n_filter[2],
							kernel_size=[3, 3],
							strides=(1, 1),
							padding="SAME",
							name='conv4' )
				conv4 = tf.layers.batch_normalization(
							conv4, 
							axis=-1,
							training=is_training )
				b4act = tf.contrib.keras.layers.LeakyReLU()
				b4act.input(conv4)

			with tf.variable_scope('Block_5'):
				conv5 = tf.layers.conv2d(
							inputs=b4act,
							filters=self.n_filter[2],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="SAME",
							name='conv5' )
				conv5 = tf.layers.batch_normalization(
							conv5, 
							axis=-1,
							training=is_training )
				b5act = tf.contrib.keras.layers.LeakyReLU()
				b5act.input(conv5)

			with tf.variable_scope('Block_6'):
				conv6 = tf.layers.conv2d(
							inputs=b5act,
							filters=self.n_filter[3],
							kernel_size=[3, 3],
							strides=(1, 1),
							padding="SAME",
							name='conv6' )
				conv6 = tf.layers.batch_normalization(
							conv6, 
							axis=-1,
							training=is_training )
				b6act = tf.contrib.keras.layers.LeakyReLU()
				b6act.input(conv6)

			with tf.variable_scope('Block_7'):
				conv7 = tf.layers.conv2d(
							inputs=b6act,
							filters=self.n_filter[3],
							kernel_size=[3, 3],
							strides=(2, 2),
							padding="SAME",
							name='conv7' )
				conv7 = tf.layers.batch_normalization(
							conv7, 
							axis=-1,
							training=is_training )
				b7act = tf.contrib.keras.layers.LeakyReLU()
				b7act.input(conv7)

			with tf.variable_scope('fully_connected'):
				fc1 = tf.contrib.layers.fully_connected(
					  inputs=b7act,
					  num_outputs=1024 )
				act = tf.contrib.keras.layers.LeakyReLU()
				act.input(fc1)

				fc2 = tf.contrib.layers.fully_connected(
					  inputs=act,
					  num_outputs=1 )
				fc2 = tf.sigmoid(x=fc2, name='output_actfcn')

		return fc2

	def downscale(self, input_x):
		arr = np.zeros([self.sr_rate, self.sr_rate, 3, 3])
		arr[:, :, 0, 0] = 1.0 / self.sr_rate ** 2
		arr[:, :, 1, 1] = 1.0 / self.sr_rate ** 2
		arr[:, :, 2, 2] = 1.0 / self.sr_rate ** 2
		weight = tf.constant(arr, dtype=tf.float32)
		downscaled = tf.nn.conv2d(
						input=input_x, 
						filter=weight, 
						strides=[1, self.sr_rate, self.sr_rate, 1], 
						padding="SAME" )
		return downscaled

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
