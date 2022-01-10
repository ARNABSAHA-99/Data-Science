#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Algorithm 

# ### Import Required Libraries and Visualize the Dataset

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#Data Reading
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()


# ### Seperate the Value of X and Y variable 

# In[2]:


#Colleccting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values


# ### Calculate the mean value of X, Y variable and the value of b0 as intercept and b1 as coefficient of X

# In[3]:


#Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Total number of values
m = len(X)

#Using the formula to calculate b1 as m and b0 as C
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#Print coefficient
print(b1, b0)


# ### Visualize the Scatter Plot and Regression Line Graph

# In[4]:


#Plotting values and Regression Line

max_x = np.max(X) + 100
min_x = np.min(X) -100

#Calcularing line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

#Ploting Line
plt.plot(x, y, color='#58b970', label= 'Regression Line')
#Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()


# ### Evaluate the Regression Line is well fitted with our actual value or not

# In[5]:


ss_t = 0 #total some of square
ss_r = 0 #total some of square of residuals
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2= 1 - (ss_r / ss_t)
print(r2)
print('This result is pretty good and distance between actual and regression value is less and percentage is almost 64%!!')


# In[ ]:




