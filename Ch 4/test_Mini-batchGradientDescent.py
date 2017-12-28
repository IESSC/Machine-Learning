# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:09:10 2017

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# In[]
# 4.2.3 Mini-batch Gradient Descent
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

lin_reg.intercept_, lin_reg.coef_

X_test = np.arange(-3,4, 0.5)
y_predict = lin_reg.coef_[0,1]*X_test**2 + lin_reg.coef_[0,0]*X_test + lin_reg.intercept_[0]
plt.plot(X_test, y_predict, "r-")
plt.axis([-3, 3, 0, 10])
plt.show()