# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:04:35 2017

@author: ASUS
"""

# Linear Regression Using the Normal Equation
import tensorflow as tf

import numpy as np
from sklearn.datasets import fetch_california_housing

# =============================================================================
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
# In[]        
housing = fetch_california_housing()

m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
tf.reset_default_graph()

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

print("By Tensor Flow: ",theta_value)    

# In[]
# Compare with pure NumPy
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print("By pure Numpy: ", theta_numpy)

# Compare with Scikit-Learn
# In[]

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print("By sklearn: ", np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])