import tensorflow as tf
import argparse
from vgg19 import vgg19

def run_model(args):
	# Define inputs and other necessary parameters
	learning_rate = 0.001
	max_epoch = 100
	batch_size = 50
	input_x = tf.placeholder(tf.float32, 
							 [None, image_size, image_size, depth], 
							 name='input' )
	target  = tf.placeholder(tf.int32, [None, args.n_class], name='target')

	# Initialize vgg
	model = vgg19(input_x, target, args.n_class, args.is_training)
	prediction = model.pred
	phi = model.phi
	loss  = model.loss
	
	#
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

	# Fetch data
	batch_img, batch_tar = fetch()
	# TensorBoard
	#merged = tf.summary.merge_all()
	
	# Saver
	saver = tf.train.Saver()
	# Initializing the variables
	init = tf.global_variables_initializer()
	
	# Session
	with tf.Session() as sess:
		#writer = tf.summary.FileWriter('board/', graph=sess.graph)
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		if args.is_training:
			for _ in range(max_epoch):
				for itr_batch in range():
					sess.run(optimizer, feed_dict={input_x: batch_img, target: batch_tar})
		else:
			#
			print "Restoring VGG19 model..."
		# Stop queue
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#parser.add_argument("dir", help="Path to training dataset")
	parser.add_argument("n_class", type=int, help="Indicate classes of the dataset.")
	parser.add_argument("-t", "--is_training", action="store_true", 
						help="Indicating mode.")
	
	args = parser.parse_args()
	run_model(args)