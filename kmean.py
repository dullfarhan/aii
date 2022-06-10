#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# In[8]:


df = pd.read_csv("data.csv")
x = df.iloc[:, [2, 3]].values
plt.scatter(df.Age, df.income)
plt.show()

# In[14]:


km = KMeans(n_clusters=4, random_state=0).fit(x)
c = km.cluster_centers_
plt.scatter(df.Age, df.income)
for y in c:
    plt.scatter(y[0], y[1], s=200, c="green")
# plt.scatter(c[1][0], c[1][1], s=200, c="red")
plt.show()

# In[ ]:


# In[ ]:




