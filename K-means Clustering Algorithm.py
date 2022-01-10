#!/usr/bin/env python
# coding: utf-8

# # K-means Clustering Algorithm

# ### Import Required Libraries and Visualize the Dataset

# In[19]:


#import required libraries 
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df = pd.read_csv("income_update.csv") #Load the dataset
print(df.shape) #Size of the dataset
df.head() #Show the few upper portion of dataset


# ### Visualize the Scatter Plot based on the Age and Income Column

# In[21]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')


# ### Declare the value of cluster and categories the data based on their cluster properties

# In[22]:


km = KMeans(n_clusters=3) #assign the value of K in km variable
y_predicted = km.fit_predict(df[['Age','Income($)']]) #Categorize the value in 3 different label
y_predicted


# ### Add the cluster column based on their cluster label

# In[23]:


df['cluster']=y_predicted #showing the cluster column in the table
df.head()


# ### Define value of three cluster's centroid value

# In[24]:


km.cluster_centers_ #print the centroids for three differnt cluster


# ### Visualize the three different in the Scatter Plot

# In[25]:


#Visualize the 3 different cluster

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()


# ### Properly Scale the Age and Income Column Values

# In[26]:


#Transform the Age and Income value within 0 to 1 for proper scaling
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[27]:


df.head()


# ### Do the same process again from define the cluster value to visualize the different cluster

# In[28]:


plt.scatter(df.Age,df['Income($)']) #Visualize the Age and Income column data


# In[29]:


km = KMeans(n_clusters=3) #assign the value of K in km variable
y_predicted = km.fit_predict(df[['Age','Income($)']]) #Categorize the value in 3 different label
y_predicted


# In[30]:


df['cluster']=y_predicted  #showing the cluster column in the table
df.head()


# In[31]:


km.cluster_centers_ #print the centroids for three differnt cluster


# In[32]:


#Visualize the 3 different cluster
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.legend()


# ### Calculate the values of Sum of Square Errors for Elbow Method

# In[34]:


#SSE= Sum of Square Errors. Determine the value of SSE for individual clusters 
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k) #for each iteration there create a new model
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_) #that is ultimately give us the SSE


# In[36]:


sse


# ### Draw the Elbow Method Plot for Proper K Value

# In[37]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:




