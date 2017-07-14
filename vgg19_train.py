import tensorflow as tf
import vgg19 as vgg
import argparse

def run_model(n_class, is_training):
	# Define inputs and other necessary parameters
	
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

		vgg = vgg.vgg19(input_x, target, args.n_class, args.is_training)
		vgg.build_model()

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
	run_model(args.is_training, args.n_class)