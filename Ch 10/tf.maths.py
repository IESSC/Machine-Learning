# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:31:26 2017

@author: ASUS
"""

import tensorflow as tf;  
import numpy as np;  

# In[]
a = np.array([[1.0, 2.0], [3.0, 4.0]])  

with tf.Session() as sess:  
    print('total mean:',  sess.run(tf.reduce_mean(a)) ) #  (1+2+3+4)/4 = 2.5
    print('mean by column:', sess.run(tf.reduce_mean(a, axis=0))  ) # [(1+3)/2, (2+4)/2] = [2, 3]
    print('mean by row:', sess.run(tf.reduce_mean(a, axis=1))  ) # [(1+2)/2, (3+4)/2] = [1.5, 3.5]

    print('total sum:',  sess.run(tf.reduce_sum(a)) ) #  (1+2+3+4) = 10
    print('sum by column:', sess.run(tf.reduce_sum(a, axis=0))  ) # [1+3, 2+4] = [4, 6]
    print('sum by row:', sess.run(tf.reduce_sum(a, axis=1))  ) # [1+2, 3+4)] = [3, 7]
    
# In[]
# tf.equal(x, y, name=None) - Returns the truth value of (x == y) element-wise. 
a = tf.constant([1, 2], tf.int32)
b = tf.constant([2, 2], tf.int32)
with tf.Session() as sess:
    print(sess.run(tf.equal(a, b))) # [False  True]

# In[]

x = tf.constant(["hehe", "haha", "hoho", "kaka"], tf.string)
y = tf.constant("hoho", tf.string)
with tf.Session() as sess:
    print(sess.run(tf.equal(x, y))) # [False False  True False]
    
# In[]
#  tf.argmax(input, axis=None, name=None, dimension=None): find index of the max value in input by axis
a = [[1,3,4,5,6]]  
b = [[1,3,4], [2,4,1]]  
  
with tf.Session() as sess:  
    print(sess.run(tf.argmax(a, 1))) # 4
    print(sess.run(tf.argmax(b, 1))) # [2,1] 
    
# In[]
#  tf.cast(x, dtype, name=None): transfer x to dtype
a = tf.convert_to_tensor(np.array([[1, 1, 2, 4], [3, 4, 5, 6]]))  
  
with tf.Session() as sess:  
    print(a.dtype)
    b = tf.cast(a, tf.float32)  
    print(b.dtype)
    
# In[]
   # tf.truncated_normal(shape, mean, stddev): mean - 2*std <= normal(mean, std) <= mean + 2*std
shape = ([2, 2, 3, 1]) # 
a = tf.truncated_normal(shape=shape, mean=0,  stddev=0.1)  
  
with tf.Session() as sess:  
    print (sess.run(a))