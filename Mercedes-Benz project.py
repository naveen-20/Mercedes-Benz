#!/usr/bin/env python
# coding: utf-8

# # Simplilearn Project

# Importing train and test data.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train_data = pd.read_csv(r'D:\Simplilearn\ML\Project submission\train\train.csv')


# In[3]:


test_data = pd.read_csv(r'D:\Simplilearn\ML\Project submission\test\test.csv')


# In[4]:


train_data.head()


# In[5]:


test_data.head()


# Getting info about the dataset.

# In[6]:


train_data.describe()


# Dropping the Id column from the dataset.

# In[7]:


train_data.drop('ID',axis=1,inplace=True)


# In[8]:


test_data.drop('ID',axis=1,inplace=True)


# In[9]:


train_data.head()


# Splitting independent variable form the train data.

# In[10]:


x_pred = train_data['y']


# In[11]:


train_data.drop('y',axis=1,inplace=True)


# In[12]:


train_data.shape


# In[13]:


train_data.head()


# Action-1: Finding the variance of the independent variable.

# In[14]:


variance = train_data.var()


# In[15]:


variance


# In[16]:


variance[variance == 0]


# Above is the column names having variance zero.
# We will drop these columns from our dataset.

# In[17]:


drop_col = ['X11','X93','X107','X233','X235','X268','X289','X290','X293','X297','X330','X347']


# In[18]:


train_data.drop(drop_col,axis=1,inplace=True)


# In[19]:


train_data.head()


# Similarly, we will remove the zero variance columns from the test data.

# In[20]:


test_data.drop(drop_col,axis=1,inplace=True)


# In[21]:


test_data.head()


# Action 2: Checking for null and unique values in train and test dataset.

# Now, we will set pandas to see all the rows.Don't hide any

# In[22]:


pd.set_option('display.max_rows',None)


# In[23]:


train_data.isnull().sum()


# Checking for null data present in test dataset.

# In[24]:


test_data.isnull().sum()


# As, we can see there are no null values present in any columns of train and test data.
# Now, lets check for the count of unique values present in train and test data.

# In[25]:


train_data.nunique()


# In[26]:


test_data.nunique()


# Action 3: Applying lable encoder to the train and test dataset to convert the categorical variable into numeric form. 

# In[27]:


from sklearn.preprocessing import LabelEncoder


# In[28]:


encoder = LabelEncoder()


# In[29]:


for col in train_data.iloc[:,0:8]:
    train_data[col] = encoder.fit_transform(train_data[col])


# In[30]:


train_data.head()


# Now, label encoding the test data.

# In[31]:


for col in test_data.iloc[:,0:8]:
    test_data[col] = encoder.fit_transform(test_data[col])


# In[32]:


test_data.head()


# Action 4: Applying dimensionality reduction on train and test dataset using Principle component             analysis.

# Standardizing the data.

# In[33]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)


# In[34]:


from sklearn.decomposition import PCA
pca = PCA(n_components= None)


# In[35]:


train_data = pca.fit_transform(train_data)


# In[36]:


test_data = pca.transform(test_data)


# In[37]:


(pca.explained_variance_ratio_)


# After looking at the explained variance ratio we can say the last 68 columns are highly correlated to each other.

# In[38]:


pca = PCA(n_components= 296)


# In[39]:


train_data = pca.fit_transform(train_data)


# In[40]:


test_data = pca.transform(test_data)


# Action 5: Predicting test data values using XGBoost. 

# In[41]:


from xgboost import XGBRegressor


# In[42]:


regressor = XGBRegressor()


# In[43]:


regressor.fit(train_data,x_pred)


# In[44]:


train_pred = regressor.predict(train_data)


# In[45]:


from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt


# In[46]:


r2_score(train_pred,x_pred)


# In[47]:


sqrt(mean_squared_error(train_pred,x_pred))


# In[48]:


test_pred = regressor.predict(test_data)


# In[49]:


test_pred

