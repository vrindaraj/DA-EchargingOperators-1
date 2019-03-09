#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING LIBRARIES

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# ## IMPORTING AND READING THE DATASET

# In[10]:


dataset = pd.read_csv("Documents/data.csv")
dataset.head()
dataset.info()
dataset.describe()
dataset.columns


# In[11]:


X  = dataset.iloc[:,1:-1]
Y =  dataset.iloc[:,11]
X["Accumulated GHG (kg)"] = X["Accumulated GHG (kg)"].apply(lambda x: float(x.split()[0].replace(',', '')))
# Y = Y.apply(lambda x: float(x.split("'")[0].replace(',', '')))
eliminate_zero = Y.mean()
Y = Y.mask(Y == 0, eliminate_zero)
X = X.values
Y = Y.values


# In[12]:


imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X)
X= imputer.transform(X)


# In[13]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[14]:


decision_tree_regression = DecisionTreeRegressor(random_state=0)
decision_tree_regression.fit(X,Y)


# ## PERFORMING DECISION TREE REGRESSION

# In[15]:


decision_prediction = decision_tree_regression.predict(X_test)
decision_prediction


# ## PREDICTION FOR TEST RESULTS

# In[8]:


decision_tree_regression.predict([[1,1,1,16,7,0.01,3,3,123,0.82]])

