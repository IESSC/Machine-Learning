
# coding: utf-8

# In[3]:

from sklearn import datasets
import pandas as pd


# In[4]:

iris = datasets.load_iris()
print(type(iris.data)) # 資料是儲存為 ndarray
print(iris.feature_names) # 變數名稱可以利用 feature_names 屬性取得


# In[5]:

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names) # 轉換為 data frame
iris_df.loc[:, "species"] = iris.target # 將品種加入 data frame
iris_df.head() # 觀察前五個觀測值


# In[7]:

iris_df


# In[ ]:



