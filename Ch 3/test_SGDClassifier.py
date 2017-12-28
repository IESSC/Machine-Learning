# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:46:50 2017

@author: hao
"""

import numpy as np
from sklearn import linear_model
# In[]
X = np.array([[1, 3], [2, 1], [-2,2],[-2,1], [1, -2],
              [5, 5], [-2, 4], [-4, 1], [4,-2], [-3,-3]])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#In[]
clf = linear_model.SGDClassifier()   #初始化隨機梯度分類法物件
clf.fit(X, Y)

result = clf.predict([[-1, 1], [4, -3]])