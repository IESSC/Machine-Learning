# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:47:41 2017
Ref: http://blog.csdn.net/mao_xiao_feng/article/details/53444333
@author: ASUS
"""
import tensorflow as tf  

a1 = tf.constant([  
        [1.0, 2.0, 3.0],  
        [5.0, 6.0, 7.0],  
        [8.0, 7.0, 6.0],
    ])   
a1=tf.reshape(a1, [1, 3, 3, 1])
  
a2 = tf.Variable(tf.random_normal([1, 3, 3, 5]))

input = a1

# In[] 
f1 =  tf.constant([  
        [1.0, 1.0],
        [1.0, 1.0],
    ]) 
f1=tf.reshape(f1, [2, 2, 1, 1])

f2 = tf.Variable(tf.random_normal([1, 1, 5, 1]))  

filter = f1
# In[]
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # input：with shape [batch, in_height, in_width, in_channels], 
    #        means [train batch of imgs, height of img, width of img, channel of img]
    #              [影像個數，影像高度，影像宽度，影像通道數(如RGB)]
    #        type should be float32 or float64
    # filter: Convolution kernel (a Tensor) with 
    #         shape [filter_height, filter_width, in_channels, out_channels]
    #               [卷積核高度，卷積核宽度，影像通道數，卷積核個數]
    #         fiter[in_channels] = input[in_channels]
    # strides: moving step of each dimension in convolution (4 dimensions)
    # padding:：{"SAME","VALID"}，
    # use_cudnn_on_gpu: bool type, using cudnn for accerlation (default: true)
    # return: feature map (a tensor) in shape of input

init = tf.initialize_all_variables()  
# In[]
with tf.Session() as sess:  
    sess.run(init)  
    print("input:"); print(sess.run(input))
    print("filter:"); print(sess.run(filter))
    print("reslut:"); print(sess.run(op))  