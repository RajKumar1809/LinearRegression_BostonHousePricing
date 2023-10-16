#!/usr/bin/env python
# coding: utf-8

# In[1]:


## importing required libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load The Boston House Pricing Dataset # 

# In[2]:


from sklearn.datasets import load_boston 


# In[3]:


boston = load_boston()


# In[4]:


boston.keys()


# In[5]:


## lets check the decription of dataset 
print(boston.DESCR)


# In[6]:


print(boston.data)


# In[7]:


print(boston.target)


# In[8]:


print(boston.feature_names)


# ## Preparing The Dataset 

# In[9]:


dataset = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[10]:


dataset.head()


# In[11]:


dataset['Price'] = boston.target


# In[12]:


dataset.head()


# In[13]:


dataset.info()


# In[14]:


#Summarizing the stats of the data
dataset.describe()


# In[15]:


## checking the missing values 
dataset.isnull().sum()


# In[16]:


import seaborn as sns
sns.pairplot(dataset)


# ## Analyzing The Correlated Features 
# 
# 

# In[17]:


## Exploratory Data Analysis 
## Correlation 
dataset.corr()


# In[18]:


plt.scatter(dataset['CRIM'], dataset['Price'])
plt.xlabel('Crime Rate')
plt.ylabel('Price')


# In[19]:


plt.scatter(dataset['RM'], dataset['Price'])
plt.xlabel('RM')
plt.ylabel('price')


# In[20]:


sns.regplot(x='RM', y ='Price', data=dataset)


# In[21]:


sns.regplot(x='LSTAT', y='Price', data=dataset)


# In[22]:


sns.regplot(x='CHAS', y='Price', data=dataset)


# In[23]:


sns.regplot(x='PTRATIO', y='Price', data=dataset)


# In[24]:


## splitting the dataset into independent and dependent features 
X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]


# In[25]:


X.head()


# In[26]:


y


# In[27]:


## Train and test split 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[28]:


X_train


# In[29]:


X_test


# In[30]:


## standardize the dataset 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()


# In[31]:


X_train = sc.fit_transform(X_train)


# In[32]:


X_test = sc.transform(X_test)


# In[33]:


import pickle
pickle.dump(sc, open('LR_scalling.pkl','wb'))


# In[34]:


X_train


# In[35]:


X_test


# ## Model Training 

# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


LR = LinearRegression()


# In[38]:


LR.fit(X_train, y_train)


# In[39]:


### Print the co-efficients and intercept 
print(LR.coef_)


# In[40]:


print(LR.intercept_)


# In[41]:


## on which params model has been trained 
LR.get_params()


# In[42]:


## Predicting with test data 
LR_pred = LR.predict(X_test)


# In[43]:


LR_pred


# ## Assumptions

# In[44]:


## plot a scatter plot for the prediction 
plt.scatter(y_test, LR_pred)


# In[45]:


## Residuals or error between actual and predictions
residuals = y_test - LR_pred


# In[46]:


residuals


# In[47]:


## plot the residuals 
sns.displot(residuals, kind='kde')


# In[48]:


## scatter plot with respect to predictions and residuals 
## uniform distribution 
plt.scatter(LR_pred, residuals)


# ## Performance Score 

# In[49]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('Mean Squared Error :',mean_squared_error(LR_pred, y_test))
print('Mean Absolute Error :',mean_absolute_error(LR_pred, y_test))
print('Square Root Of Mean Squared Error :',np.sqrt(mean_squared_error(LR_pred, y_test)))


# # R Square And Adjusted R Square
# Formula
# ## R^2 = 1 - SSR/SST 
# ## R^2 = coefficient of determination 
# ## SSR = sum of squares of residuals 
# ## SST = total sum of squares 

# In[50]:


from sklearn.metrics import r2_score 
score = r2_score(LR_pred, y_test)
print(score)


# ## Adjusted R2 = 1 -[(1-R2)*(n-1)/(n-k-1)]
# Where : 
# R2 : The R^2 of the model n : the number of observations k: the number of predictor variables 

# In[51]:


## Display The adjusted R-squared 
1 - (1-score)*(len(y_test)-1)/(len(y_test) - X_test.shape[1]-1)


# # New Data Prediction 

# In[52]:


boston.data[0].reshape(1, -1)


# In[53]:


## transformation of new data 
sc.transform(boston.data[0].reshape(1, -1))


# In[54]:


LR.predict(sc.transform(boston.data[0].reshape(1, -1)))


# In[55]:


# Pickling The Model File For Deployment 


# In[56]:


import pickle


# In[57]:


pickle.dump(LR, open('regmodel.pkl', 'wb'))


# In[58]:


pickled_model = pickle.load(open('regmodel.pkl', 'rb'))


# In[59]:


## Prediction 
pickled_model.predict(sc.transform(boston.data[0].reshape(1, -1)))


# In[ ]:





# In[ ]:





# In[ ]:




