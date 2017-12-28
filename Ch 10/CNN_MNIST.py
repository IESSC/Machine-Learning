# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:38:08 2017
ref: https://www.tensorflow.org/get_started/mnist/pros
@author: ASUS
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 設定參數
logs_path = 'TensorBoard/'
n_features = x_train.shape[1]
n_labels = y_train.shape[1]

with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, shape=[None, n_features])/255.        
    # reshape image to tensor format 
    x_image = tf.reshape(xs, [-1, 28, 28, 1]) 
    
with tf.name_scope('Label'):
    ys = tf.placeholder(tf.float32, shape=[None, n_labels])
with tf.name_scope('Parameter'):
    keep_prob = tf.placeholder(tf.float32)

# In[]
def compute_accuracy(v_xs, v_ys):
    global y_pred
    y_pre = sess.run(y_pred, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        #  tf.argmax(input, axis=None, name=None, dimension=None):
        #    Returns the index of the max value in input by axis (0: row, 1: column)
        #  tf.equal(x, y, name=None) 
        #    Returns the truth value of (x == y) element-wise.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

    # Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
        # tf.truncated_normal(shape, mean, stddev): generate normal dist.
        # normal(0, 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
        # tf.constant(value, dtyp, shape, name, verify_shape): Creates a constant tensor
    return tf.Variable(initial)

def conv2d(x, W): # Convolution 卷積
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
        # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        # input：with shape [batch, in_height, in_width, in_channels], 
        #        means [train batch of imgs, height of img, width of img, channel of img]
        #              [影像個數，影像高度，影像宽度，影像通道數(如RGB)]
        #        type should be float32 or float64
        # filter: Convolution kernel (a Tensor) with 
        #         shape [filter_height, filter_width, in_channels, out_channels]
        #               [filter高度，filter宽度，影像通道數，filter個數]
        #         filter[in_channels] = input[in_channels]
        # strides: moving step of each dimension in convolution (4 dimensions)
        # padding:：{"SAME","VALID"}，
        # use_cudnn_on_gpu: bool type, using cudnn for accerlation (default: true)
        # return: feature map (a tensor) in shape of input
        # ref: tf.nn.conv2d.py
    
def max_pool_2x2(x): # Pooling 池化
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
        # tf.nn.max_pool(value, ksize, strides, padding, name=None)
        # value: input of pool (feature map) with shape [batch, height, width, channels]
        # ksize: size of pool window with 4 dimensions, 
        #        ex. [1, height, width, 1], due to not pool in batch和channels, set both dimensions as 1
        # strides: (similar to convolution) moving step of window in each dimension, ex. [1, stride,stride, 1]
        # padding:(similar to concolution) , {'VALID', 'SAME'}
        # return: (a Tensor), in shape of [batch, height, width, channels]
        # ref: tf.nn.max_pool.py
    
# In[]
# first convolution layer
with tf.name_scope('First-Convolution-Layer'):
    # assume a 5*5 fiter is inputted to derive 32 features (or filters) 
    # ref to https://ithelp.ithome.com.tw/articles/10187282
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32]) # to avoid negative
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # [None, 28, 28, 32]
        # tf.nn.crelu(features, name=None) = max(features, 0)
        # ref to: tf.nn.actfun.py
    
    h_pool1 = max_pool_2x2(h_conv1) #[None, 14, 14, 32] --> from 28*28 to 14*14

# In[]
# Second convolution layer 
with tf.name_scope('Second-Convolution-Layer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #[None, 14, 14, 64]
    h_pool2 = max_pool_2x2(h_conv2) #[None, 7, 7, 64] --> from 14*14 to 7*7

# In[]
# Densely Connected Layer
with tf.name_scope('Densely-Connected-Layer'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #[None, 1024]
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #[None, 1024]

# In[] 
# Readout Layer
with tf.name_scope('Readout-Layer'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #[None, 10]

# In[]
with tf.name_scope('Cross-Entropy'):
        # ref to: tf.maths.py
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_pred), reduction_indices=[1])) 
        # sum by column, total mean
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
# In[]  initialization
    
# 啟動 InteractiveSession
sess = tf.InteractiveSession()
#sess = tf.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# In[]
# visual output
#merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

# In[]
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        summary = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])
        #writer.add_summary(summary, i)
        #writer.flush()    
        print('step %d, training accuracy %g' % (i, summary)) 
