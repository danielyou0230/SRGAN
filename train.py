import tensorflow as tf
import argparse
from srgan import srgan
from PIL import Image
from utils import read_and_decode, save_image

tfr_path = "tfrecords/"
tfrecord = ["train.tfrecord", "test.tfrecord"]
model_vgg = ""
batch_size = 50
learning_rate = 1e-4
image_size = 96
max_iter = 10000

def train():
	input_x = tf.placeholder(tf.float32, 
							 [None, image_size, image_size, depth], 
							 name='input' )

	
	is_training = tf.placeholder(tf.bool, [])

	#model = srgan(input_x=input_x, is_training=is_training)
	
	#define 
	net_g = SRGAN_g(input_x=input_x, is_train=True, reuse=False)
	net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
	_,logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

	#resize

	#define loss
	d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
	d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
	d_loss = d_loss1 + d_loss2

	g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
	mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
	vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

	g_loss = mse_loss + vgg_loss + g_gan_loss

	g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
	d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)


	#
	global_step = tf.Variable(0, name='global_step', trainable=False) 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	g_optimizer = optimizer.minimize(
					loss=model.g_loss,
					global_step=global_step,
					var_list=model.g_variables,
					name='g_optimizer' )

	d_optimizer = optimizer.minimize(
					loss=model.d_loss,
					global_step=global_step,
					var_list=model.d_variables,
					name='g_optimizer' )
	
	# Generate tfrecord file name
	tfr = []

	# Load training data
	img, label = read_and_decode(tfr[0])
	batch_img, _ = tf.train.shuffle_batch([img, label],
					batch_size=batch_size, capacity=1000,
					min_after_dequeue=100,
					allow_smaller_final_batch=True )
	# Load testing data
	t_img, t_label = read_and_decode(tfr[1])
	test_img, _ = tf.train.shuffle_batch([t_img, t_label],
					batch_size=800, capacity=800,
					min_after_dequeue=0,
					allow_smaller_final_batch=True )

	#merged = tf.summary.merge_all()
	
	# Saver
	saver = tf.train.Saver()
	# Initializing the variables
	init = tf.global_variables_initializer()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True

	with tf.Session() as sess:
		#sess = tf.Session(config=config)
		#writer = tf.summary.FileWriter('board/', graph=sess.graph)
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		# Restore VGG model
		var = tf.global_variables()
		vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
		saver = tf.train.Saver(vgg_var)
		saver.restore(sess, vgg_model)

		# Restore SRGAN (if argument -t is not given)
		if not args.is_training:
			saver = tf.train.Saver()
			saver.restore(sess, "path to chk file")

		# Train SRGAN
		for progress in range(max_iter)
			sess.run(, feed_dict={input_x: batch_img, is_training: True})

		# Validation
		sess.run(, feed_dict={input_x: batch_img, is_training: False})
		save_image()

		# Save latest model
		save_ckpt = saver.save(sess, "model.ckpt")
		print "Model saved in file: {:s}".format(save_ckpt)
		# Stop queue
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("tfpath", type=int, help="Indicate classes of the dataset.")
	parser.add_argument("-t", "--is_training", action="store_true", 
						help="Indicating mode.")
	
	args = parser.parse_args()
	train(args)