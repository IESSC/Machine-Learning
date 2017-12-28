# Chapter 2 End to End Machine Learning
import os

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing/"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH
#-------------------------------------------------------------
# In[ Step 1. load csv data file
import pandas as pd
def load_housing_data(housing_path= HOUSING_URL):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path, sep=',')

housing = load_housing_data()

# In[ Step 1.1 show data information
housing.head()
housing.info()

#-------------------------------------------------------------
# In[ Step 1.2 show data distribution
housing["ocean_proximity"].value_counts()

housing.describe()
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

#-------------------------------------------------------------
# In[Step 2. Create a Test Set
import numpy as np

# In[]
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

#-------------------------------------------------------------
# In[ step 2.1 another method for spliting data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")

#-------------------------------------------------------------
# In[ step 2.2 stratified sampling by median_income categories 
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
housing["income_cat"].value_counts() / len(housing)

# In[ remove the new attribute from train and test sets
for s in (strat_train_set, strat_test_set):
    s.drop(["income_cat"], axis=1, inplace=True) # axis=1: column not row

#--------------------------------------------------------------    
# In[ Step 3. Visualizing Geographical Data
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")

# alpha: density of data point 
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# s(radius of each circle): population; c(color): price; cmap: predefined color map
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, 
             s=housing["population"]/100, label="population", 
             c="median_house_value", cmap=plt.get_cmap("jet"), 
             colorbar=True,)  
plt.legend()

#---------------------------------------------------------------
# In[ Step 3.1 Looking for Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(15, 12))
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

#-----------------------------------------------------------------
# In[ Step 3.2 Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]   #每人房屋數
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] #每屋臥房數 
housing["population_per_household"]=housing["population"]/housing["households"] #每人人口比
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_values = strat_train_set["median_house_value"].copy()

#------------------------------------------------------------------
# In[Step 4. Data Cleaning
# Step 4.1 prepocessing missing values of attribute 

from sklearn.preprocessing import Imputer
housing_num = housing.drop("ocean_proximity", axis=1)  # select partial data from housing

  # try to fill median for missing data
imputer = Imputer(strategy="median")  # new Imputer object  
imputer.fit(housing_num)  # apply Imputer to fill mediam for each attribute

X = imputer.transform(housing_num) # transfer from dataframe to array
housing_tr = pd.DataFrame(X, columns=housing_num.columns) 

#--------------------------------------------------------------------
# In[ Step 4.2 Handling Text and Categorical Attributes
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()  # linear encoding
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

# In[]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

# In[] better encoding for categorical data
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)

#---------------------------------------------------------------------
# In[ Step 4.3 Combine Custom Transformers by a Class (step 3.2)
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]           #每人房屋數 array
        population_per_household = X[:, population_ix] / X[:, household_ix] #每人人口比 array
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]          #每屋臥房數 array
            return np.c_[X, rooms_per_household, population_per_household, 
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]  # arrays to object
        
attr_adder = CombinedAttributesAdder()  # new an Comhousing.valuesbinedAttributesAdder object
#housing_extra_attribs = attr_adder.transform()
#-------------------------------------------------------------------
# In[ Step 4.4 Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# In[] try 1
num_pipeline = Pipeline([ ('imputer', Imputer(strategy="median")),
                          ('attribs_adder', CombinedAttributesAdder()),
                          ('std_scaler', StandardScaler()),])
                 
housing_num_tr = num_pipeline.fit_transform(housing_num)
#==============================================================================

#==============================================================================
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing_num_tr[attributes], figsize=(12, 8))
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)
#==============================================================================

# In[] get attribute data by assigning attribute names
#-------------------------------------------------------------------
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


# In[]
from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# pipeline for processing numerical data as in steps 4.1 and 4.3 
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                         ])

# pipeline for processing categorical data as in step 4.2
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('label_binarizer', LabelBinarizer()),
                         ])

full_pipeline = FeatureUnion(
        transformer_list=[("num_pipeline", num_pipeline),
                          ("cat_pipeline", cat_pipeline),
                          ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape

#----------------------------------------------------------------
# In[ Step 5. Select and Train a Model
# Step 5.1 using linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_values)

# In[]
from sklearn.metrics import mean_squared_error
# =============================================================================
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_values, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

# =============================================================================
# In[] Step 5.2 using decision tree regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_values)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_values, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# =============================================================================
# In[  
# Step 5.2.1 Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_values,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
[rmse_scores, rmse_scores.mean(), rmse_scores.std()]

# In[] Step 5.2.2 using random forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_values)
forest_reg_predictions = forest_reg.predict(housing_prepared)
forest_reg_mse = mean_squared_error(housing_values, forest_reg_predictions)
forest_reg_rmse = np.sqrt(forest_reg_mse)

# In[]
from sklearn.externals import joblib
joblib.dump(forest_reg, "forest_reg_model.pkl")
# and later...
forest_reg_loaded = joblib.load("forest_reg_model.pkl")
#--------------------------------------------------------------------------
# In[ Step 5.3 using Fine-Tune Model
from sklearn.model_selection import GridSearchCV
param_grid = [ {'n_estimators': [3, 10, 30], 
                'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 
                 'n_estimators': [3, 10], 
                 'max_features': [2, 3, 4]},]

forest_reg = RandomForestRegressor()

# cv: cross validation
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_values)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# In[] Step 6 Evaluate Your System on the Test Set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)