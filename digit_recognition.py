import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data
import mnistHandwriting
train_MNIST = mnistHandwriting.MNISTexample(0,1000)
test_MNIST = mnistHandwriting.MNISTexample(1000, 2000,bTrain=True)


X_training = [train_MNIST[i][0] for i in range(len(train_MNIST))]
Y_training = [train_MNIST[i][1] for i in range(len(train_MNIST))]

X_testing = [test_MNIST[i][0] for i in range(len(test_MNIST))]
Y_testing = [test_MNIST[i][1] for i in range(len(test_MNIST))]



# Define model parameters
learning_rate = 0.001
training_epochs = 100

number_of_inputs = 784 # 28 * 28 size
number_of_outputs = 10

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50


# models

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases


with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 10))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)




# training
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    # training
    for epoch in range(training_epochs):

        session.run(optimizer, feed_dict={X: X_training, Y: Y_training})
        
        if epoch % 5 == 0:
            training_cost = session.run([cost], feed_dict={X: X_training, Y:Y_training})
            testing_cost = session.run([cost], feed_dict={X: X_testing, Y:Y_testing})

            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))
        

    # traning finished
    final_training_cost = session.run(cost, feed_dict={X: X_training, Y: Y_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_testing, Y: Y_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))
