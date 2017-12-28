# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:47:06 2017

@author: hao
"""
model_data = [1, 2, 6, 4, 2, 4, 7, 5,9]
test_data = [6, 9, 4]

model_label = ["taipei", "tainan", "kaohsiung", "tainan", "roma", "paris"]
test_label = ["tainan",'taipei', 'kaohsiung']


# In[]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()  # linear encoding
encoder.fit(model_data)               # LabelEncoder()
encoder.classes_
encoder.transform(test_data)          # array([3], dtype=int64)

encoder.fit_transform(model_data)
encoder.inverse_transform([0,1,3,2])  # array([1, 2, 6, 4])

# In[]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()  # linear encoding

encoder.fit(model_label)               # LabelEncoder()
encoder.classes_
result = encoder.transform(test_label)          # array([0, 3])
encoder.inverse_transform([0,1,3,2])   # array(['kaohsiung', 'paris', 'tainan', 'roma'], dtype='<U9')
encoder.fit_transform(model_label)     # array([4, 3, 0, 3, 2, 1], dtype=int64)

# In[]
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
encoder.fit(model_data)
encoder.classes_  #array([1, 2, 4, 6])
encoder.transform(test_data) # array([[1, 0, 0, 0], [0, 0, 0, 1]])

# In[]
model_data2 = [1, 2.3, 6.4, 2.7, 8]
test_data2 = [3.3, 4.9]
from sklearn.preprocessing import StandardScaler
import numpy as np

ss = StandardScaler()
result = ss.fit(np.array(model_data2).reshape(-1,1))
print(ss.mean_)
result = ss.transform(np.array(test_data2).reshape(-1,1))
print(result)
ss.fit_transform(np.array(model_data2).reshape(-1,1))
