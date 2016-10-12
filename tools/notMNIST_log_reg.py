########### Batch gradient descent #########
# i.e averaging derivates over entire training dataset

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys, os

image_size = 28
num_labels = 10

DATA_DIR = "/home/deeplearning/work/tensorflow/datasets/" + sys.argv[1]

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

######################## REFORMATER #######################
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

##########################################################

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


################ simple logistic regression with batch grad descent ###############################

x = tf.placeholder(tf.float32,[None,784])  # None means any size

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)
y_= tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.InteractiveSession()
tf.initialize_all_variables().run()

for i in range(500):  # epochs in bracket
	batch_xs, batch_ys = train_dataset, train_labels
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	print("epochs: ",i)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: valid_dataset, y_: valid_labels}))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_dataset, y_: test_labels}))


