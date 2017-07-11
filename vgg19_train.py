import tensorflow as tf
import vgg19 as vgg

def run_model():
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

		vgg = vgg.vgg19()
		vgg.build()
		
		# Stop queue
		coord.request_stop()
		coord.join(threads)