# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:26:59 2017

@author: ASUS
"""
# Creating and running a graph
# !conda install -c https://conda.anaconda.org/jjhelmus tensorflow
import tensorflow as tf
# In[]
# Linear Regression Using the Normal Equation
    
import numpy as np
from sklearn.datasets import fetch_california_housing

# ====== modify: fetch_california_housing ====================================================
#         tarobj = tarfile.open(
#                 mode="r:gz",
#                 name=archive_path)
#         fileobj = tarobj.extractfile(
#                 'CaliforniaHousing/cal_housing.data')
# 
#         cal_housing = np.loadtxt(fileobj, delimiter=',')
#         fileobj.close()
#         tarobj.close()
#         remove(archive_path)
# =============================================================================
        
housing = fetch_california_housing()

m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# In[]
tf.reset_default_graph()

X = tf.constant(housing_data_plus_bias, dtype=tf.float64, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float64, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

print(theta_value)    

# In[]
# Using Batch Gradient Descent

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

# In[]
class tfSession():
    def __init__(self, input_x, input_y):
        tf.reset_default_graph()
        self.X = tf.constant(input_x, dtype=tf.float32, name="X")
        self.y = tf.constant(input_y, dtype=tf.float32, name="y")
        self.theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
        self.y_pred = tf.matmul(self.X, self.theta, name="predictions")
        self.error = self.y_pred - self.y
        self.mse = tf.reduce_mean(tf.square(self.error), name="mse")
        
#-------------
    def run_tfSession(self, training_op):

        n_epochs = 1000
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                if epoch % 100 == 0:
                    print("Epoch", epoch, "MSE =", self.mse.eval())
                    sess.run(training_op)
                    best_theta = self.theta.eval()

        print("Best theta:")
        print(best_theta)
#--------------

# In[]
# Manually computing the gradients
  # init_tfSession()    

tfs = tfSession(input_x = scaled_housing_data_plus_bias, 
                input_y = housing.target.reshape(-1, 1))

learning_rate = 0.01
  #--------------
gradients = 2/m * tf.matmul(tf.transpose(tfs.X), tfs.error)
training_op = tf.assign(tfs.theta, tfs.theta - learning_rate * gradients)

  #-------------
tfs.run_tfSession(training_op)
#--------------
# In[]
# Using autodiff
tfs = tfSession(input_x = scaled_housing_data_plus_bias, 
                input_y = housing.target.reshape(-1, 1))

learning_rate = 0.01
#--------------
gradients = tf.gradients(tfs.mse, [tfs.theta])[0]
training_op = tf.assign(tfs.theta, tfs.theta - learning_rate * gradients)

  #-------------
tfs.run_tfSession(training_op)

# In[]
# Using a GradientDescentOptimizer
tfs = tfSession(input_x = scaled_housing_data_plus_bias, 
                input_y = housing.target.reshape(-1, 1))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(tfs.mse)

tfs.run_tfSession(training_op)
#--------------

# In[]
# Using a momentum optimizer
tfs = tfSession(input_x = scaled_housing_data_plus_bias, 
                input_y = housing.target.reshape(-1, 1))

learning_rate = 0.01
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.25)
training_op = optimizer.minimize(tfs.mse)

tfs.run_tfSession(training_op)


