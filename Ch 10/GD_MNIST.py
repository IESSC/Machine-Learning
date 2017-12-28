# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:38:08 2017

@author: ASUS
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

# In[]
learning_rate = 0.5

x_dim = train_img.shape[1]
y_dim = train_label.shape[1]

x = tf.placeholder(tf.float32, [None, x_dim]) # None: any dimension
W = tf.Variable(tf.zeros([x_dim, y_dim])) # W: x_dim x y_dim
b = tf.Variable(tf.zeros([y_dim]))
y = tf.nn.softmax(tf.matmul(x, W) + b) # y = Wx +b

y_ = tf.placeholder(tf.float32, [None, y_dim]) 

    # cross_entropy = -sum(y_ * log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
    # http://colah.github.io/posts/2015-09-Visual-Information/    
    # ref to: tf.nn.actfun.py

    # using backpropagation algorithm with min cross_entropy
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # http://colah.github.io/posts/2015-08-Backprop/

init = tf.global_variables_initializer()
# In[]

sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))