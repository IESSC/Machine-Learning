# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:56:13 2017

@author: ASUS
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

ouput = tf.multiply(input1, input2)

# In[]
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))