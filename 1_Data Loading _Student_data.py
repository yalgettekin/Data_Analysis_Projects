#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


# Read CSV dataset 
df = pd.read_csv("Students_data.csv")


# In[8]:


# Print first few entries 
df.head()


# In[9]:


# Print last few entries
df.tail()


# In[10]:


# Check number of missing values
df.isnull().sum()


# In[12]:


# Retrieve data types
df.dtypes


# In[14]:


# Retrieve statistical description
df.describe() # default use


# In[15]:


# include all attributes
df.describe(include = "all")


# In[16]:


# Retrive data set summary
df.info()


# In[ ]:




