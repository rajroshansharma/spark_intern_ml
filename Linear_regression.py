#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML

# ## Task 1 : Prediction Using Supervised ML

# ### Predict the percentage of an student based on the no. of study hours.
# 

# In[1]:


#Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Reading data from URL
url = "http://bit.ly/w-data"
stu_data = pd.read_csv(url)
stu_data.head()


# In[4]:


#First 5 dataset
stu_data.tail()


# In[5]:


#Plotting hours studied vs percentage score graph
stu_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours Studied vs Percentage Scored')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Scored')  
plt.show()


# In[6]:


X = stu_data.iloc[:, :-1].values  
y = stu_data.iloc[:, 1].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[8]:



line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y,color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.plot(X, line);
plt.show()


# In[9]:


#Testing data
print(X_test) 
#Prediction of scores
y_pred = regressor.predict(X_test)


# In[10]:


#Comparison between actual and predicted data
comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
comp


# In[11]:


print("Training Score ",regressor.score(X_train,y_train))
print("Testing Score ",regressor.score(X_test,y_test))


# In[12]:


comp.plot(kind='bar',figsize=(6,6),color=('blue','red'))
plt.show()


# In[13]:


# Percentage scored when studied for 9.25 hours
hours = np.array(9.25)
print("No. of hours studied = {}".format(hours))
hours = hours.reshape(-1,1)
score_pred = regressor.predict(hours)
print("Predicted Score = {}".format(score_pred[0]))


# In[14]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('Explained Variance Score:',metrics.explained_variance_score(y_test,y_pred))


# In[ ]:




