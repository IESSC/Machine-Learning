# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:10:55 2017
Ref: http://blog.csdn.net/mao_xiao_feng/article/details/53453926
@author: ASUS
"""

import tensorflow as tf  
  
# An image with two channels in shape of 4*4
a=tf.constant([  
        [1.0, 2.0, 3.0, 4.0],  
        [5.0, 6.0, 7.0, 8.0],  
        [8.0, 7.0, 6.0, 5.0],  
        [4.0, 3.0, 2.0, 1.0],
    ])  

b=tf.constant([  
        [[1.0, 2.0, 3.0, 4.0],  
         [5.0, 6.0, 7.0, 8.0],  
         [8.0, 7.0, 6.0, 5.0],  
         [4.0, 3.0, 2.0, 1.0]],
         
        [[4.0, 3.0, 2.0, 1.0],  
         [8.0, 7.0, 6.0, 5.0],  
         [1.0, 2.0, 3.0, 4.0],  
         [5.0, 6.0, 7.0, 8.0]]  
    ])  

# In[]
# reshape tensor a
# r=tf.reshape(a,[1, 4, 4, 1])  
r=tf.reshape(b,[1, 4, 4, 2])  

pooling = tf.nn.max_pool(r, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID') 
    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    # value: input of pool (feature map) with shape [batch, height, width, channels]
    # ksize: size of pool window with 4 dimensions, 
    #        ex. [1, height, width, 1], due to not pool in batch和channels, set both dimensions as 1
    # strides: moving step of fiter window in each dimension, ex. [1, stride,stride, 1]
    # padding: {'VALID', 'SAME'}
    # return: (a Tensor), in shape of [batch, height, width, channels] 
# In[]

with tf.Session() as sess:
    print("image:"); print (sess.run(r))  
    
    print("reslut:"); print (sess.run(pooling))

# In[]
pooling = tf.nn.max_pool(r, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  

with tf.Session() as sess:  
    print("reslut:"); print (sess.run(pooling))