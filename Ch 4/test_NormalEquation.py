# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:27:32 2017

@author: ASUS
"""
    
import numpy as np
X = np.array( [[1, 0],[2, 3],[3, 1], [4,7]]) # two variables with 4 smaples

y = 5 + 3* X + np.random.randn(len(X), 1)

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
