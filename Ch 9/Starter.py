# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:29:53 2017

@author: ASUS
"""

# Creating and running a graph
# !conda install -c https://conda.anaconda.org/jjhelmus tensorflow
import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2  # 3*3*4 + 4 + 2 = 42

print(f)
# In[]
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

print(result)

#graph = tf.Graph()
#with graph.as_default():
#    x2 = tf.Variable(2)
#    
# In[]
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15
    
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15


# In[]
# Placeholder nodes
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 3))

Y = X + 5

with tf.Session() as sess:
     Y1 = Y.eval(feed_dict={X: [[1, 2, 3]]})
     Y2 = Y.eval(feed_dict={X: [[4, 5, 6], [7, 8, 9]]})

print(Y1)
print(Y2)    

# In[]
import tensorflow as tf  
  
x1 = tf.placeholder(tf.float32, [2, 3])  
x2 = tf.placeholder(tf.float32, [2, 3])  
  
x3 = tf.placeholder(tf.float32)  
x4 = tf.placeholder(tf.float32)  
  
# maxtrix  
y1 = tf.matmul(x1, tf.transpose(x2))  
# numberic multiply 
y2 = tf.multiply(x3, x4)  
  
with tf.Session() as sess:  
    #feed_dict
    print (sess.run(y1, feed_dict ={x1:[[1, 2, 3], [4, 5, 7]], 
                                    x2:[[6, 5, 4], [1, 7, 8]]}))  
    print (sess.run(y2, feed_dict = {x3:3, x4:4}))  
    print (sess.run(y2, feed_dict = {x3:[4], x4:[5]}))  
