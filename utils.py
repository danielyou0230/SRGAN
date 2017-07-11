import tensorflow as tf 
from PIL import Image
import os
import numpy as np
import argparse

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_converter(file_path, tf_data):
	buff = []
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		n_file = sum(os.path.isfile(os.path.join(file_path, itr_dir)) \
					for itr_dir in os.listdir(file_path))
		# Converting data to tfrecord
		for itr_file in os.listdir(file_path):
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
	print "{:s}: {:d} files".format(file_path, n_file)

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbosity", action="store_true",
						help="show info in each directory")
	parser.add_argument("path", help="Path to images.")
	parser.add_argument("output_name", \
						help="Output file name. (file extension will be added automatically)")

	args = parser.parse_args()
	if not os.path.isdir('tfrecords'):
		os.makedirs('tfrecords')

	if not args.output_name.endswith('.tfrecord'):
		file_name = args.output_name + '.tfrecord'
	else:
		file_name = args.output_name

	data_converter(path, file_name)