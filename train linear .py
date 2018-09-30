
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\parag\\Desktop\\train.csv")


# In[3]:


df.head()


# In[6]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
type(correlation_values)


# In[10]:


correlation_values[["Id", "SalePrice"]]


# In[11]:


X = df[["OverallQual", "TotalBsmtSF", "GrLivArea", "GarageArea"]]


# In[12]:


y = df['SalePrice']


# In[15]:


from sklearn.model_selection import train_test_split as tts


# In[16]:


X_train, X_test, y_train, y_test = tts(X, y, test_size= 0.3, random_state= 42)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[19]:


reg = LinearRegression()


# In[20]:


reg.fit(X_train, y_train)


# In[21]:


y_pred = reg.predict(X_test)


# In[24]:


reg.score(X_test, y_test)


# In[25]:


from sklearn.metrics import r2_score


# In[26]:


r2_score(y_test, y_pred)

