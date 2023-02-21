#!/usr/bin/env python
# coding: utf-8

# # GRIP: THE SPARKS FOUNDATION
# # DATA Science and business Analytics intership
# # Author: Sourabh Mishra 

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)


# In[37]:


data.head()   # to view the first 5 rows of the data


# In[38]:


data.info()   # to see if there are any missing values


# In[39]:


data.describe()  # to see the summary statistics of the data


# In[40]:


data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[41]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[42]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[43]:


y_pred = regressor.predict(X_test)


# In[44]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[45]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line)
plt.title('Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[ ]:





# In[46]:


hours = [[9.25]]
predicted_score = regressor.predict(hours)
print("Number of study hours: {}".format(hours))
print("Predicted Score = {}".format(predicted_score[0]))

