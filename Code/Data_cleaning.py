#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd


# ## Importing Dataset

# In[2]:


train = pd.read_csv('Documents/datam.csv')


# ## Printing the head values of the dataset

# In[3]:


train.head()


# In[4]:


print (train['Charge Fee(in$)'])


# ## Checking the Null Values in the dependent variable

# In[5]:


print (train['Charge Fee(in$)'].isnull())


# ## Total Number of columns for each row

# In[6]:


train.apply(lambda x: x.count(), axis=1)


# In[7]:


train.isnull()


# In[8]:


missing_values = ["n/a", "na", "--"]
train = pd.read_csv("G:\project\PRoject final yr\data.csv", na_values = missing_values)


# In[9]:


train.isnull()


# ## To find the sum of null values in each column

# In[10]:


print (train.isnull().sum())


# In[11]:


print (train.isnull().values.any())


# In[12]:


print (train.isnull().sum().sum())


# ## To show the specific row and column which has null values

# In[13]:


null_columns=train.columns[train.isnull().any()] 


# In[14]:


print(train[train["No Of Sessions"].isnull()][null_columns])


# In[15]:


median_value=train["No Of Sessions"].median()
train["No Of Sessions"]=train["No Of Sessions"].fillna(median_value)


# In[16]:


print(train[train["Accumulated Sessions"].isnull()][null_columns])


# In[17]:


median_value=train["Accumulated Sessions"].median()
train["Accumulated Sessions"]=train["Accumulated Sessions"].fillna(median_value)


# In[18]:


print(train[train["No. of Ports"].isnull()][null_columns])


# In[19]:


median_value=train["No. of Ports"].median()


# In[20]:


train["No. of Ports"]=train["No. of Ports"].fillna(median_value)


# In[21]:


print(train[train["Energy(kWh)"].isnull()][null_columns])


# In[22]:


meadian_value=train["Energy(kWh)"].median()
train["Energy(kWh)"]=train["Energy(kWh)"].fillna(median_value)


# In[23]:


print(train[train["GHG_savings(kg)"].isnull()][null_columns])


# In[24]:


meadian_value=train["GHG_savings(kg)"].median()
train["GHG_savings(kg)"]=train["GHG_savings(kg)"].fillna(median_value)


# In[25]:


print(train[train["Charge Fee(in$)"].isnull()][null_columns])


# In[26]:


train["Charge Fee(in$)"]=train["Charge Fee(in$)"].fillna(0)


# ## To check whether the dataset has no null values

# In[27]:


print (train.isnull().sum())

