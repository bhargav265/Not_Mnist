from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

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
image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1,image_size*image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset,  labels
train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
train_subset = 10000
batch_size = 128
n_hidden1 = 1024
n_hidden2 =  512
n_hidden3 = 256
n_hidden4 = 128
n_hidden5 = 64

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.placeholder(tf.float32,shape = (batch_size,image_size*image_size))
  tf_train_labels = tf.placeholder(tf.float32,shape = (batch_size,num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.

  #hidden layer 1
  weights1 = tf.get_variable("weights1", shape=[image_size*image_size, n_hidden1],
           initializer=tf.contrib.layers.xavier_initializer())
  biases1 = tf.Variable(tf.zeros([n_hidden1]))
  hidden_layer1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  keep_prob = tf.placeholder("float")
  hidden_layer_drop1 = tf.nn.dropout(hidden_layer1, keep_prob)

  #hidden layer 2
  
  weights2 = tf.get_variable("weights2", shape=[n_hidden1, n_hidden2],
           initializer=tf.contrib.layers.xavier_initializer())
  biases2 = tf.Variable(tf.zeros([n_hidden2]))
  hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer_drop1,weights2) + biases2)
  hidden_layer_drop2 = tf.nn.dropout(hidden_layer2, keep_prob)


  #hidden layer 3
  
  weights3 = tf.get_variable("weights3", shape=[n_hidden2, n_hidden3],
           initializer=tf.contrib.layers.xavier_initializer())
  biases3 = tf.Variable(tf.zeros([n_hidden3]))
  hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer_drop2,weights3) + biases3)
  hidden_layer_drop3 = tf.nn.dropout(hidden_layer3, keep_prob)

  #hidden layer 4
  weights4 = tf.get_variable("weights4", shape=[n_hidden3, n_hidden4],
           initializer=tf.contrib.layers.xavier_initializer())
  biases4 = tf.Variable(tf.zeros([n_hidden4]))
  hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer_drop3,weights4) + biases4)
  hidden_layer_drop4 = tf.nn.dropout(hidden_layer4, keep_prob)

  #hidden layer 5
  weights5 = tf.get_variable("weights5", shape=[n_hidden4, n_hidden5],
           initializer=tf.contrib.layers.xavier_initializer())
  biases5 = tf.Variable(tf.zeros([n_hidden5]))
  hidden_layer5 = tf.nn.relu(tf.matmul(hidden_layer_drop4,weights5) + biases5)
  hidden_layer_drop5 = tf.nn.dropout(hidden_layer5, keep_prob)

  #output layer

  weights = tf.get_variable("weights", shape=[n_hidden5, num_labels],
           initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.Variable(tf.zeros([num_labels]))



  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(hidden_layer_drop5, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + (1e-3)*tf.nn.l2_loss(weights1)  + (1e-3)*tf.nn.l2_loss(weights2) + (1e-3)*tf.nn.l2_loss(weights3) + (1e-3)*tf.nn.l2_loss(weights4) + (1e-3)*tf.nn.l2_loss(weights5)+ (1e-3)*tf.nn.l2_loss(weights))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.9)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  
  #validation prediction
  valid_relu1 = tf.nn.relu( tf.matmul(tf_valid_dataset,weights1) + biases1)
  valid_relu2 = tf.nn.relu( tf.matmul(valid_relu1,weights2) + biases2)
  valid_relu3 = tf.nn.relu( tf.matmul(valid_relu2,weights3) + biases3)
  valid_relu4 = tf.nn.relu( tf.matmul(valid_relu3,weights4) + biases4)
  valid_relu5 = tf.nn.relu( tf.matmul(valid_relu4,weights5) + biases5)
  valid_prediction = tf.nn.softmax( tf.matmul(valid_relu5,weights) + biases)
  
  #test prediction
  test_relu1 = tf.nn.relu( tf.matmul(tf_test_dataset,weights1) + biases1)
  test_relu2 = tf.nn.relu( tf.matmul(test_relu1,weights2) + biases2)
  test_relu3 = tf.nn.relu( tf.matmul(test_relu2,weights3) + biases3)
  test_relu4 = tf.nn.relu( tf.matmul(test_relu3,weights4) + biases4)
  test_relu5 = tf.nn.relu( tf.matmul(test_relu4,weights5) + biases5)
  test_prediction = tf.nn.softmax( tf.matmul(test_relu5,weights) + biases)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
num_steps = 10001

#train_dataset = train_dataset[:500, :]
#train_labels = train_labels[:500]

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict = feed_dict)
    if (step % 500 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, batch_labels))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))