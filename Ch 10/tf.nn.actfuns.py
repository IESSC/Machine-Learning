# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:03:23 2017
ref: http://www.jianshu.com/p/e3a79eac554f
@author: ASUS
"""
import tensorflow as tf

a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])

with tf.Session() as sess:
    #----------------    
    b = tf.nn.relu(a)
        # tf.nn.relu(features, name = None) = max(features, 0)
    print ('tf.nn.relu: ', sess.run(b))   
    
    #------------------
    b = tf.nn.softplus(a)
        # tf.nn.softplus(features, name = None) = log(exp(features) + 1)
    print ('tf.nn.softplus: ', sess.run(b))  
    
    #------------------
    b = tf.nn.dropout(a, 0.5, noise_shape = [1,4])
        #tf.nn.dropout(x, keep_prob, noise_shape = None, seed = None, name = None) 
    print ('tf.nn.dropout: ', sess.run(b))   
        
    #------------------
    b = tf.nn.sigmoid(a)
         #tf.sigmoid(features, name = None) = 1 / (1 + exp(-features))
    print ('tf.nn.sigmoid: ', sess.run(b))        
        

