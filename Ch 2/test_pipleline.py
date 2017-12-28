# -*- coding: utf-8 -*-
"""
Transformer in scikit-learn - some class that have fit and transform method, or fit_transform method.

Predictor - some class that has fit and predict methods, or fit_predict method.

Pipeline is just an abstract notion, it's not some existing ml algorithm. 
Often in ML tasks you need to perform sequence of different transformations 
(find set of features, generate new features, select only some good features) 
of raw dataset before applying final estimator.
"""
# In[] Original steps

"""
s1 = Step1() 
s2 = Step2()
s3 = Step3()

model1 = s1.fit_transform(Xtrain)
model2 = s2.fit_transform(model1)
model3 = s3.fit_predict(model2)

test1 = s1.fit_transform(Xtest)
test2 = s2.fit_transform(test1)
test3 = s3.fit_predict(test2)

"""
# In[] Pipeline steps

"""
pipeline = Pipeline([
    ('s1', Step1()),
    ('s2', Step2()),
    ('s3', Step3()),
])
model3 = pipeline.fit(Xtrain).predict(Xtrain)

test3 = pipeline.predict(Xtest)

""""

# http://pbpython.com/pandas-list-dict.html

model_data1 = [(-1, 2, 6.4, 2.7),
               (1, 2),
               (3,   2,   6, 1.7)]  # model_data1 is a list
model_label = ['A','B','C','D']     # model_label1 is a list
test_data1 = [3.3, 4.9]

import pandas as pd

# transform lists to a dataframe
df = pd.DataFrame.from_records(model_data1, columns=model_label) 

# In[]

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")  # new Imputer object
imputer.fit(df)                       # fill the dataframe with median 
dfm = imputer.transform(df)           # transfer the dataframe to array dfm

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()              
ss.fit(dfm)                           # standardize the array dfm 
dfms = ss.transform(dfm)              

# In[]
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline ([
        ('imputer', Imputer(strategy="median")),
        ('normalize', StandardScaler()),
        ])

result = pipeline.fit_transform(df)