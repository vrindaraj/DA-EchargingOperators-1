#!/usr/bin/env python
# coding: utf-8

# ## Dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset and Describing it

# In[2]:


df = pd.read_csv("Documents/data.csv")
df.head()
df.info()
df.describe()
df.columns


# ## Visualization of dataset

# In[3]:


sns.pairplot(df)


# ## Plot with respect to the dependent variable

# In[4]:


sns.distplot(df['Charge Fee(in$)'])


# In[5]:


df.corr()


# In[6]:


X  = df.iloc[:,1:-1]
Y =  df.iloc[:,11]
X["Accumulated GHG (kg)"] = X["Accumulated GHG (kg)"].apply(lambda x: float(x.split()[0].replace(',', '')))
# Y = Y.apply(lambda x: float(x.split("'")[0].replace(',', '')))
eliminate_zero = Y.mean()
Y = Y.mask(Y == 0, eliminate_zero)
X = X.values
Y = Y.values


# ## Preprocessing  Independent Variables

# In[7]:


imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X)
X= imputer.transform(X)


# ## Splitting training set and testing set

# In[9]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# ## Performing Simple Linear Regression

# In[10]:


linear_regression = LinearRegression()
linear_regression.fit(X_train,Y_train)


# ## Predicting the Test Results

# In[11]:


prediction = linear_regression.predict(X_test)


# In[12]:


prediction


# ## pediction for a same set of data

# In[13]:


linear_regression.predict([[1,1,1,16,7,0.01,3,3,123,0.82]])

