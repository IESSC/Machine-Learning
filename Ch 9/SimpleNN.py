# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:45:54 2017

@author: hao
"""
import tensorflow as tf  
import numpy as np  
  
# In[]
def add_layer(input_data, input_size, output_size, activation_function = None):  

    #generate random values with input_size 
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))  
    #generate a column of 0.1
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)  
    
    #input_data * weights + biases  
    Wx_plus_b = tf.matmul(input_data, Weights) + biases  
    if activation_function is None:  
        output = Wx_plus_b  
    else:  
        output = activation_function(Wx_plus_b)  
    return output  

# In[]
#generate a list with 1000 elements in -1 to 1 
x_data = np.linspace(-1, 1, 1000)[:, np.newaxis]  
noise = np.random.normal(0, 0.05, x_data.shape)  
y_data = np.square(x_data) - 0.5 + noise # target value 
  
xs = tf.placeholder(tf.float32, [None, 1])  
ys = tf.placeholder(tf.float32, [None, 1])  
  
#set default layer  
layer1 = add_layer(xs, 1, 10, tf.nn.relu)  
prediction = add_layer(layer1, 10, 1, None)  

loss = tf.reduce_mean(tf.square(ys - prediction))  
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  
  
init = tf.global_variables_initializer()  

# In[]  
with tf.Session() as sess:  
    sess.run(init)  
      
    for step in range(1001):  
        
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})  
        if step % 50 == 0:  
            print(step, sess.run(loss, feed_dict = {xs: x_data, ys:y_data}))
            
            
            