from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys, os

image_size = 28
num_labels = 10

DATA_DIR = "/home/deeplearning/work/tensorflow/datasets/" + sys.argv[1]


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  return dataset, labels


pickle_file = os.path.join(DATA_DIR, 'notMNIST.pickle')

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


###########################################################

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x=tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_relu1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
h_relu2 = tf.nn.relu(h_conv2)
h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
h_fc1_relu = tf.nn.relu(h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession()
tf.initialize_all_variables().run()

index = range(train_dataset.shape[0])
print(len(index))

for i in range(20000):
	random_index = np.random.choice(index, 1500) # returns list of random index # define mini-batch size for SGD
	batch_xs, batch_ys = train_dataset[random_index], train_labels[random_index]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.50})
	
	if i%100 == 0:
		valid_accuracy=sess.run(accuracy, feed_dict={x: valid_dataset, y_: valid_labels, keep_prob: 1.0})
    		print("step %d, validation accuracy %g"%(i, valid_accuracy))
  

print("test accuracy %g"%(sess.run(accuracy, feed_dict={x: test_dataset, y_: test_labels, keep_prob: 1.0})))

