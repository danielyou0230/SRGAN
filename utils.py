import tensorflow as tf 
from PIL import Image
import os
import numpy as np
import argparse
import sys

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecords(args):
	dataset_name = file_path if not file_path.endswith('/') else file_path[:-1]
	n_test  = int(len(all_files) * args.percentage)
	all_files = list()
	for itr in os.listdir(args.file_path):
		if itr.endswith('.jpg')
			all_files.append(itr)

	print "{:s}: {:d} files, {:d}%  saved for testing."\
	.format(args.file_path, len(all_files), args.percentage)
	
	random.shuffle(all_files)
	print "Exporting training data as \"train.tfrecords\" ... ", 
	data_converter(file_path=args.file_path, 
				   tf_data="{:s}_train.tfrecords".format(dataset_name), 
				   idx_list=all_files[:-1 * n_test])
	print "Done"
	print "Exporting testing data as \"test.tfrecords\" ... ", 
	data_converter(file_path=args.file_path, 
				   tf_data="{:s}_test.tfrecords".format(dataset_name), 
				   idx_list=all_files[-1 * n_test: ])
	print "Done"

def data_converter(file_path, tf_data, idx_list=None):
	buff = []
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		n_file = sum(os.path.isfile(os.path.join(file_path, itr_dir)) \
					for itr_dir in os.listdir(file_path)) \
					if not idx_list else len(idx_list)
		# Converting data to tfrecord
		load_list = os.listdir(file_path) if not idx_list else idx_list
		for itr_file in load_list:
			if itr_file.endswith('.jpg'):
				img_path = file_path + itr_file
				img = Image.open(img_path)
				img_raw = img.tobytes()
				# Stream data to the converter
				example = tf.train.Example(features=tf.train.Features(
				feature=
				{ 
					"label"  : _int64_feature(1),
					"img_raw": _bytes_feature(img_raw)
				} ))
				converter.write(example.SerializeToString())
			else:
				continue
	print "{:s}: {:s} ({:d} files)".format(tf_data, file_path, n_file)

def read_and_decode(filename, img_size=128, depth=1, normalize=False):
	if not filename.endswith('.tfrecords'):
		print "Invalid file \"{:s}\"".format(filename)
		return
	else:
		data_queue = tf.train.string_input_producer([filename])

		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(data_queue) 
		features = tf.parse_single_example(serialized_example,
				   features={
							 'label'   : tf.FixedLenFeature([], tf.int64),
							 'img_raw' : tf.FixedLenFeature([], tf.string),
							})

		img = tf.decode_raw(features['img_raw'], tf.uint8)
		img = tf.reshape(img, [img_size, img_size, depth])
		if normalize:
		# Normalize the image
			img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
		label = tf.cast(features['label'], tf.int32)
		label_onehot = tf.stack(tf.one_hot(label, n_classes))
		return img, label_onehot

def save_image(batch, tag, epoch):
	output_path = "result_{:s}_{:d}".format(tag, epoch)
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	
	for itr in range(batch_size):
		# convert to image format
		# save image

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbosity", action="store_true",
						help="show info in each directory")
	parser.add_argument("file_path", help="Path to images.")
	parser.add_argument("percentage", type=int, help="Percentage for testing.")
	#parser.add_argument("output_name", \
	#help="Output file name. (file extension will be added automatically)")

	args = parser.parse_args()
	if not os.path.isdir(args.file_path):
		print "Invalid input path: {:s}".format(args.file_path)
		sys.exit()

	if not os.path.isdir('tfrecords'):
		os.makedirs('tfrecords')

	generate_tfrecords(args)
	print "Script completed! Check the files under tfrecords/"