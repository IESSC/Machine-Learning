# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:54:46 2017

@author: ASUS
"""
import tensorflow as tf

# In[]
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# In[]
import numpy as np
from sklearn.datasets import fetch_california_housing
   
housing = fetch_california_housing()

m, n = housing.data.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# In[]
import numpy.random as rnd
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

# In[]
# tensorboard --logdir tf_logs/
# http://127.0.0.1:6006/#scalars
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# In[]
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
              #  save_path = saver.save(sess, logdir+"model.ckpt")
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
  #  save_path = saver.save(sess, logdir+"model_final.ckpt")
    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()
print("Best theta:")
print(best_theta)

# In[]
#with tf.Session() as sess:
#    saver.restore(sess, logdir+"model_final.ckpt")
