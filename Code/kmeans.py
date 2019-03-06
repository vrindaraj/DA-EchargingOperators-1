#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset

# In[35]:


train = pd.read_csv("Documents/datam.csv")
test = pd.read_csv("Documents/data.csv")


# ## Load the train and test datasets to create two DataFrames

# In[10]:


print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())


# In[11]:


print("***** Train_Set *****")
print(train.describe())
print("\n")
print("***** Test_Set *****")
print(test.describe())


# ## Preprocessing Dataset and Variables

# In[12]:


print(train.columns.values)


# In[13]:


train.isna().head()


# In[14]:


test.isna().head()


# In[15]:


print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())


# In[16]:


train = train.applymap(lambda Charge_Type: 1 if Charge_Type == True else Charge_Type)
train = train.applymap(lambda Charge_Type: 0 if Charge_Type == False else Charge_Type)


# ## Fill missing values with mean column values in the Data set

# In[17]:


train.fillna(train.mean(), inplace=True)


# In[18]:


print(train.isna().sum())


# In[19]:


train[["Charge Fee(in$)", "Charge time (minutes)"]].groupby(['Charge Fee(in$)'], as_index=False).mean().sort_values(by='Charge time (minutes)', ascending=False)


# In[20]:


train[["Charge Fee(in$)", "Energy(kWh)"]].groupby(['Charge Fee(in$)'], as_index=False).mean().sort_values(by='Energy(kWh)', ascending=False)


# ## Pie Chart for the Variables

# In[21]:


g = sns.FacetGrid(train, col='Charge_Type')
g.map(plot.hist, 'Charge Fee(in$)', bins=20)


# In[22]:


grid = sns.FacetGrid(train, col='Charge_Type', row='Charge_demand', height=2.5, aspect=1.6)
grid.map(plot.hist, 'Charge Fee(in$)',alpha=1, bins=10)
grid.add_legend();


# ## Let's investigate if you have non-numeric data left

# In[23]:


train.info()


# In[24]:


X = np.array(train.drop(['Charge_Type'], 1).astype(float))


# In[25]:


y = np.array(train['Charge_Type'])


# In[26]:


train.info()


# In[27]:


kmeans = KMeans(n_clusters=2)


# In[28]:


kmeans.fit(X)


# ## Prediction

# In[29]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# ## Data clustering

# In[30]:


data =  pd.read_csv("Documents/data.csv")
cluster_X = data.iloc[:,1:]
cluster_X["Accumulated GHG (kg)"] = cluster_X["Accumulated GHG (kg)"].apply(lambda x: float(x.split()[0].replace(',', '')))
cluster_X = cluster_X.values
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(cluster_X)
cluster_X = imputer.transform(cluster_X)


# ## Clustering

# In[31]:


K_Means = KMeans(3)
K_Means.fit(cluster_X)


# ## pedictions

# In[32]:


cluster_prediction = K_Means.fit_predict(cluster_X)
prediction_dataset = data.copy()
prediction_dataset['cluters'] = cluster_prediction


# ## kmeans_clustering

# In[33]:


plot.scatter(prediction_dataset['Accumulated_Energy(MWh)'],prediction_dataset['Charge Fee(in$)'],c=prediction_dataset['cluters'],cmap='rainbow')
plot.xlim(0,50)
plot.ylim(0,100)
plot.show()

