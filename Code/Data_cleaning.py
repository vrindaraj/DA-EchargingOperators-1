#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[29]:


import pandas as pd


# In[30]:


import numpy as np


# ## Importing Dataset

# In[31]:


df = pd.read_csv('G:\project\PRoject final yr\data.csv')


# ## Printing the head values of the dataset

# In[32]:


df.head()


# In[33]:


print (df['Charge Fee(in$)'])


# ## Checking the Null Values in the dependent variable

# In[34]:


print (df['Charge Fee(in$)'].isnull())


# ## Total Number of columns for each row

# In[35]:


df.apply(lambda x: x.count(), axis=1)


# In[36]:


df.isnull()


# In[37]:



missing_values = ["n/a", "na", "--"]
df = pd.read_csv("G:\project\PRoject final yr\data.csv", na_values = missing_values)


# In[38]:


df.isnull()

# ## To find the sum of null values in each column

# In[40]:


print (df.isnull().sum())


# In[41]:


print (df.isnull().values.any())


# In[42]:


print (df.isnull().sum().sum())


# In[43]:


null_columns=df.columns[df.isnull().any()] 
  


# ## To show the specific row and column which has null values

# In[44]:


print(df[df["No Of Sessions"].isnull()][null_columns])


# In[45]:


df["No Of Sessions"].fillna("28", inplace = True)


# In[46]:


print(df[df["Accumulated Sessions"].isnull()][null_columns])


# In[47]:


df["Accumulated Sessions"].fillna("3559", inplace = True)


# In[48]:


print(df[df["No. of Ports"].isnull()][null_columns])


# In[49]:


df["No. of Ports"].fillna("17", inplace = True)


# In[50]:


print(df[df["Energy(kWh)"].isnull()][null_columns])


# In[51]:


df["Energy(kWh)"].fillna("32", inplace = True)


# In[52]:


print(df[df["GHG_savings(kg)"].isnull()][null_columns])


# In[53]:


df["GHG_savings(kg)"].fillna("0", inplace = True)


# In[54]:


print(df[df["Charge Fee(in$)"].isnull()][null_columns])


# In[55]:


df["Charge Fee(in$)"].fillna("0", inplace = True)


# ## To check whether the dataset has no null values

# In[56]:


print (df.isnull().sum())

