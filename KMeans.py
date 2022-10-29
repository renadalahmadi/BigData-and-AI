#!/usr/bin/env python
# coding: utf-8

# # KMeans Clustering

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # Dog Horse KMeans Challenge

# In[62]:


# Import Data
df_dh = pd.read_csv("https://raw.githubusercontent.com/gumdropsteve/datasets/master/dog_or_horse.csv")
# We picked variables that we need
X = df_dh[['height', 'weight']]
df_dh.head()


# 1- Standardize the data

# In[80]:


# Scale Data
scaler = StandardScaler()

# Fit & transform data.
scaled_df = scaler.fit_transform(X)
scaled_df


# 2- Create an Elbow Plot to determine the number of clusters

# In[64]:


# Create Elbow Plot

# The elbow method depends on WCSS which stands for Within Cluster Sum of Squares

wcss = []
# Note: We are using K-mean++ to avoid the random initialization trap 
# Note: We are creating a plot of the WCSS for upto 10 clusters using the for loop
# The measurement we are using is the inertia 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# - From the plot we can see that the optimal number of clusters in 2. 

# 3- Apply the K-Means Clustering model

# In[65]:


# Apply KMeans and Plot KMeans Results and Actual Results
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 123)
y_kmeans = kmeans.fit_predict(scaled_df)


# In[66]:


print(y_kmeans)


# In[67]:


# Cluster centeriods
print (kmeans.cluster_centers_)


# In[68]:


# assign the y_kmeans to a new column in the dataset
df_dh['kmean_prid']= y_kmeans
df_dh


# In[69]:


# replace dog with 1 and horse with 0 to help us caluclate the Accuracy Score
df_dh['type'] = df_dh['type'].map({'dog': 1, 'horse': 0})


# In[84]:


# we will need this to get the score of the kmean model
actualType = df_dh['type']
actualType


# In[71]:


# Calculate the number of correct predictions
(df_dh['type'] == df_dh['kmean_prid']).value_counts()


# In[78]:


y_kmeans


# 4- Plot the clusters including the centroid for each of the clusters as defined by K-Means
# 

# In[79]:



plt.scatter(scaled_df[y_kmeans == 0, 0],scaled_df[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(scaled_df[y_kmeans == 1, 0], scaled_df[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'brown', label = 'Centroids')

plt.title('Dog Horse KMeans Challenge')
plt.xlabel('height')
plt.ylabel('weight')
plt.legend()
plt.show()


# 5- the accuracy score if true labels are known

# In[87]:


# Compute Accuracy Score of KMean Labels with True Labels

score = round(accuracy_score(y_kmeans, actualType), 4)
print('Accuracy of Kmeans :{0:f}'.format(score))


# In[90]:


# another approach could be used to the model is to split the data as usual into : 
X2 = df_dh[['height', 'weight']]
y2 = df_dh['type']

X_train, X_test,y_train,y_test = train_test_split(X2,y2,test_size=0.20,random_state=70)


# In[92]:


# fit the model only to the train set: 
k_means2 = KMeans(n_clusters = 2, init = 'k-means++', random_state = 123)
k_means2.fit(X_train)


# In[94]:


# the model labels: 
print(k_means2.labels_[:])
print(y_train[:])


# In[96]:


# Compute Accuracy Score of KMean Labels

score = accuracy_score(y_test,k_means2.predict(X_test))
print('Accuracy:{0:f}'.format(score))


# # Seattle Weather KMeans Challenge

# In[144]:


df_sea = pd.read_csv("https://raw.githubusercontent.com/gumdropsteve/datasets/master/seattle_weather_1948-2017.csv")
X_sea = df_sea[['tmax', 'tmin']]
df_sea.head()


# In[145]:


# Scale Data
scaler = StandardScaler()

# Fit & transform data.
scaled_df = scaler.fit_transform(X_sea)
scaled_df


# In[146]:


# Create Elbow Plot

# The elbow method depends on WCSS which stands for Within Cluster Sum of Squares

wcss = []
# Note: We are using K-mean++ to avoid the random initialization trap 
# Note: We are creating a plot of the WCSS for upto 10 clusters using the for loop
# The measurement we are using is the inertia 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X_sea)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the plot we can see that the optimal number of clusters in 4, but our target is 2 so we should pick 2.


# In[147]:


# Apply KMeans and Plot KMeans Results and Actual Results
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 123)
y_kmeans = kmeans.fit_predict(scaled_df)


# In[166]:



plt.scatter(scaled_df[y_kmeans == 0, 0],scaled_df[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')

plt.scatter(scaled_df[y_kmeans == 1, 0], scaled_df[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'brown', label = 'Centroids')

plt.title('Seattle Weather KMeans Challenge')
plt.xlabel('tmax')
plt.ylabel('tmin')
plt.legend()
plt.show()


# In[165]:


y_kmeans


# In[150]:


# assign the y_kmeans to a new column
df_sea['kmean_prid']= y_kmeans
df_sea


# In[152]:


# replace True with 1 and False with 0 to help us caluclate the Accuracy Score
df_sea.replace({False: 0, True: 1}, inplace=True)


# In[153]:


# Calculate the number of correct predictions
(df_sea['rain'] == df_sea['kmean_prid']).value_counts()


# In[162]:


df_sea.isnull().sum()
df_sea['rain'].unique()


# In[ ]:


df_sea.dropna()


# In[ ]:


actualvalues = df_sea['rain'].iloc[:10000,:]
actualvalues 


# In[ ]:


# Compute Accuracy Score of KMean Labels with True Labels
score = round(accuracy_score(y_kmeans, actualvalues), 2)
print('Accuracy of Kmeans :{0:f}'.format(score))


# # Random Blob KMeans Challenge
# 
# - You dont have true labels for this data so this is truly an unsupervised dataset
# - The blobs are randomly generated every time you run the cell and their characteristics are:
#     - 2000-4000 data points
#     - 10-30 blobs created

# In[178]:


df_blob = pd.DataFrame(make_blobs(random.randint(2000,4000), centers=random.randint(10,30))[0])
df_blob.plot(kind="scatter", x=0, y=1, title="Blobs", figsize=(12,10));


# In[184]:


# Scale Data
scaler = StandardScaler()

# Fit & transform data.
scaled_df = scaler.fit_transform(df_blob)
scaled_df


# In[186]:


# The elbow method depends on WCSS which stands for Within Cluster Sum of Squares

wcss = []
# Note: We are using K-mean++ to avoid the random initialization trap 
# Note: We are creating a plot of the WCSS for upto 10 clusters using the for loop
# The measurement we are using is the inertia 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X_sea)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[187]:


# Apply KMeans and Plot KMeans Results and Actual Results
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 123)
y_kmeans = kmeans.fit_predict(scaled_df)


# In[188]:


# Plot the clusters 

plt.scatter(scaled_df[y_kmeans == 0, 0],scaled_df[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(scaled_df[y_kmeans == 1, 0], scaled_df[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'brown', label = 'Centroids')

plt.title('# Random Blob KMeans Challenge')
plt.legend()
plt.show()

