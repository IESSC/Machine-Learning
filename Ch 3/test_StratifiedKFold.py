# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:48:49 2017

@author: hao

K次交叉驗證，初始採樣分割成K個子樣本，一個單獨的子樣本被保留作為驗證模型的數據，其他K-1個樣本用來訓練。
交叉驗證重複K次，每個子樣本驗證一次，平均K次的結果或者使用其它結合方式，最終得到一個單一估測。
這個方法的優勢在於，同時重複運用隨機產生的子樣本進行訓練和驗證，每次的結果驗證一次，10次交叉驗證是最常用的。
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

# In[]
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9,10], [11,12]])
y = np.array([0, 1, 1, 1, 1, 0])

# In[]
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)  

# In[]
for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, y_tarin = X[train_index], y[train_index]
   X_test, y_test = X[test_index], y[test_index]
