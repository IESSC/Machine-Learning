# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:30:23 2017

@author: ASUS
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# In[]
mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

# In[]

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

# In[]
from sklearn.metrics import accuracy_score

y_pred = list(dnn_clf.predict(X_test))
accuracy = accuracy_score(y_test, y_pred)
accuracy

# In[]
from sklearn.metrics import log_loss

y_pred_proba = list(dnn_clf.predict_proba(X_test))
log_loss(y_test, y_pred_proba)

dnn_clf.evaluate(X_test, y_test) #{'accuracy': 0.98269999, 'global_step': 40000, 'loss': 0.074661769}
