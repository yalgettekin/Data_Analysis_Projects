#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("listings.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# Removing the Duplicates if any¶
# 

# In[7]:


df.duplicated().sum()
df.drop_duplicates(inplace = True)


# Check null values

# In[8]:


df.isnull().sum()


# Drop unnecessary columns

# In[9]:


df.drop(["name", "host_name","neighbourhood_group","last_review"], axis = 1,inplace =True)


# In[10]:


df.head()


# Rreplace the 'reviews per month' by zero
# 

# In[11]:


df.fillna({"reviews_per_month":0}, inplace = True)
df.reviews_per_month.isnull().sum()


# In[12]:


df.head()


# Remove the NaN values from the dataset¶
# 

# In[13]:


df.isnull().sum()
df.dropna(how="any", inplace=True)
df.info()


# Examine Continous Variables
# 

# In[14]:


df.describe()


# Print all the columns names
# 

# In[15]:


df.columns


# Get Correlation between different variables
# 

# In[16]:


df_numeric = df.select_dtypes(include=[int, float])

correlation_matrix = df_numeric.corr(method="kendall")

plt.figure(figsize=(15, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
df.columns


# In[23]:


df[["price","minimum_nights","availability_365"]].corr()


# In[17]:


df.shape


# # Data Visualization

# In[18]:


plt.figure(figsize=[8,8],clear = True, facecolor = "#ABB2B9")
df["room_type"].value_counts().plot.pie(autopct = "%1.3f%%", shadow = True);


# In[19]:


df.hist(edgecolor="black", linewidth=1.2, figsize=(30, 30));


# In[ ]:




