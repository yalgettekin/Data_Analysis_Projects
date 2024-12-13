#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np


# In[6]:


df = pd.read_csv("auto.csv")


# In[7]:


df.head()


# In[8]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)


# In[9]:


df.columns = headers
df.columns


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# In[12]:


df["normalized-losses"].unique()


# In[13]:


df.replace("?",np.nan,inplace = True)
df.head(5)


# In[14]:


df.isnull().sum()


# # Dealing with the missing data
# 1. Drop Data
#     1. Drop the row
#     2. Drop the column
#     
# 2. Replace the data
#     1. Replace it by mean
#     2. Replace it by frequency
#     3. Replace it based on other functions
#     

# <b> Replace by mean: </b>
# * "normalized-losses": 41 missing data, replace them with mean
# * "stroke": 4 missing data, replace them with mean
# * "bore": 4 missing data, replace them with mean
# * "horsepower": 2 missing data, replace them with mean
# * "peak-rpm": 2 missing data, replace them with mean
# 
# 

# <b>Replace by frequency:</b>
# * "num-of-doors": 2 missing data, replace them with "four".
#     * Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur

# <b> Drop the whole row:</b>
# * "price": 4 missing data, simply delete the whole row
#     * Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you.

# <h4>Calculate the mean value for the "normalized-losses", "stroke", "bore", "horsepower" and "peak-rpm"  columns </h4>
# 

# In[15]:


avg_norm = df["normalized-losses"].astype("float").mean(axis=0)
avg_stroke = df["stroke"].astype("float").mean(axis=0)
avg_bore= df["bore"].astype("float").mean(axis=0)
avg_horsep = df["horsepower"].astype("float").mean(axis=0)
avg_peakrpm = df["peak-rpm"].astype("float").mean(axis=0)


# In[22]:


print("Average of the normalized-losses:", avg_norm)
print("Average of stroke:", avg_stroke)
print("Average of bore:", avg_bore)
print("Avearge of horsepower:", avg_horsep)
print("Average of peak-rpm", avg_peakrpm)


# <h4>Replace "NaN" with the mean value in the "normalized-losses", "stroke", "bore", "horsepower" and "peak-rpm" columns</h4>
# 

# In[23]:


df.replace(np.nan, avg_norm, inplace = True)
df.replace(np.nan, avg_stroke, inplace = True)
df.replace(np.nan, avg_bore, inplace = True)
df.replace(np.nan, avg_horsep, inplace = True)
df.replace(np.nan, avg_peakrpm, inplace = True)


# In[24]:


df.head()


# <b>Replace by frequency:</b>
# 

# To see which values are present in a particular column, we can use the ".value_counts()" method:

# In[26]:


df["num-of-doors"].value_counts()


# "num-of-doors": 2 missing data, replace them with "four".
#    * Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur

# We can also use the ".idxmax()" method to calculate the most common type automatically:

# In[27]:


df["num-of-doors"].value_counts().idxmax()


# Lets replace nan values by "four"

# In[29]:


df["num-of-doors"].replace(np.nan, "four", inplace=True)


# drop all rows that do not have price data:

# In[31]:


df.dropna(subset=["price"],axis=0, inplace= True)
df.reset_index(drop = True, inplace = True)


# In[32]:


df.head()


# In[ ]:




