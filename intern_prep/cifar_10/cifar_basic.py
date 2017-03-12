
import numpy as np
import tensorflow as tf
import os, sys
from six.moves import cPickle as pickle
from PIL import Image

class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_dir = sys.argv[1]


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def load_pickle(filename):
	with open(filename, 'rb') as f:
		save = pickle.load(f)
		raw_data = save['data']
		raw_labels = save['labels']
		del save  # hint to help gc free up memory
	return raw_data, raw_labels


def check_data_img(tensor, labels, img_name):
	img = Image.fromarray(tensor[5], 'RGB')
	img.save( img_name + '.jpg')
	print "Label: ", class_list[np.argmax(labels[5])]
	#img.show()


def reformat_data(raw_data, raw_labels):
	#image_major_data = raw_data.reshape(10000, 3072).astype(np.uint8)
	channel_major_data = raw_data.reshape(-1, 3, 1024).astype(np.uint8)
	tensor = channel_major_data.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # transpose was done to view image. currently effect on training not known

	labels_array = np.array(raw_labels).astype(np.int)	
	labels_1h = np.zeros((len(labels_array), 10))
	labels_1h[np.arange(len(raw_labels)), labels_array] = 1.0
	print tensor.shape, labels_1h.shape
	return tensor, labels_1h


def raw_data_collect(data_dir, section):
	raw_labels = []
	raw_data = []
	for batch in os.listdir(os.path.join(data_dir, section)):
		raw_batch_data, raw_batch_labels = load_pickle(os.path.join(data_dir, section, batch))
		raw_data.append(raw_batch_data)
		raw_labels = raw_labels + raw_batch_labels		

	raw_data = np.array(raw_data).reshape(-1, 3072)	
	print raw_data.shape, len(raw_labels)
	return raw_data, raw_labels


def normalize(data):
	normalized_data = (data.astype(np.float32) - 128.0) / 255.0
	return normalized_data	

def separation(shuffled_dataset, shuffled_labels, n_validate = 0):
	
	train_data = shuffled_dataset[n_validate:]
	train_labels = shuffled_labels[n_validate:]

	valid_data = shuffled_dataset[:n_validate]
	valid_labels = shuffled_labels[:n_validate]

	return train_data, train_labels, valid_data, valid_labels

def shuffle(train_data, train_labels):

	permutation = np.random.permutation(train_labels.shape[0])
	shuffled_dataset = train_data[permutation]
	shuffled_labels = train_labels[permutation]
	return shuffled_dataset, shuffled_labels

def network_model(x_image, h_fc1_prob):

	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
	h_relu1 = tf.nn.relu(h_conv1)
	h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
	h_relu2 = tf.nn.relu(h_conv2)
	h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

	W_fc1 = weight_variable([8 * 8 * 64, 1024])
	b_fc1 = bias_variable([1024])
	
	h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
	h_fc1_relu = tf.nn.relu(h_fc1)	
	h_fc1_drop = tf.nn.dropout(h_fc1_relu, h_fc1_prob)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	return y_conv



if __name__ == '__main__':
	
	train_data, train_labels = raw_data_collect(data_dir, 'train')
	test_data, test_labels = raw_data_collect(data_dir, 'test')

	train_tensor, train_labels = reformat_data(train_data, train_labels)
	test_tensor, test_labels = reformat_data(test_data, test_labels)

	train_tensor, train_labels = shuffle(train_tensor, train_labels)
	train_tensor, train_labels, valid_tensor, valid_labels = separation(train_tensor, train_labels, 5000) # validation datasize is added here

	# check_data_img(train_tensor, train_labels, "sample_train")
	# check_data_img(test_tensor, test_labels, "sample_test")
	# check_data_img(valid_tensor, valid_labels, "sample_valid")

	train_tensor = normalize(train_tensor)
	valid_tensor = normalize(valid_tensor)
	test_tensor = normalize(test_tensor)



	sess = tf.Session()

	x_image = tf.placeholder(tf.float32, [None, 32, 32, 3])
	target = tf.placeholder(tf.float32, [None, 10])

	h_fc1_prob = tf.placeholder(tf.float32)
	output = network_model(x_image, h_fc1_prob)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
	train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
	
	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	minibatch = 20
	sess.run(tf.initialize_all_variables())

	for epoch in range(10):
		train_tensor, train_labels = shuffle(train_tensor, train_labels)

		for k in xrange(0, len(train_labels), minibatch):	
			batch_input, batch_output = train_tensor[k : k + minibatch] , train_labels[k : k + minibatch] 
			sess.run(train_step, feed_dict = {x_image: batch_input, target: batch_output, h_fc1_prob: 0.60 })

			valid_accuracy = sess.run(accuracy, feed_dict={x_image: valid_tensor, target: valid_labels, h_fc1_prob: 1.0})
			print "epoch %d, validation accuracy %g"%(epoch, valid_accuracy)

	print "test accuracy %g"%(sess.run(accuracy, feed_dict={x_image: test_tensor, target: test_labels, h_fc1_prob: 1.0}))

