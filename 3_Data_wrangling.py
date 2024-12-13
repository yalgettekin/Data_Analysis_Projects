#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("auto_new.csv")


# In[5]:


df.head()


# <b> Fix the data types 
# 

# The goal is to making sure that all data is in the correct fromat
# <p><b>.dtype()</b> to check the data type</p>
# <p><b>.astype()</b> to change the data type</p>

# In[8]:


df.dtypes


# As you can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, the numerical values 'bore' and 'stroke' describe the engines, so you should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. You have to convert data types into a proper format for each column using the "astype()" method.

# df[[‘attribute1’, ‘attribute2’, ...]] =
# df[[‘attribute1’, ‘attribute2’,
# ...]].astype(‘data_type’)
# #data_type can be int, float, char, etc.
# 

# In[9]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")


# In[11]:


df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")


# In[12]:


df.dtypes


# ## Data Standardization
# <p>
# You usually collect data from different agencies in different formats.
# (Data standardization is also a term for a particular type of data normalization where you subtract the mean and divide by the standard deviation.)
# </p>
#     
# <b>What is standardization?</b>
# <p>Standardization is the process of transforming data into a common format, allowing the researcher to make the meaningful comparison.
# </p>
# 
# <b>Example</b>
# <p>Transform mpg to L/100km:</p>
# <p>In your data set, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit. Assume you are developing an application in a country that accepts the fuel consumption with L/100km standard.</p>
# <p>You will need to apply <b>data transformation</b> to transform mpg into L/100km.</p>

# <p>Use this formula for unit conversion:<p>
# L/100km = 235 / mpg
# <p>You can do many mathematical operations directly using Pandas.</p>

# In[13]:


df["city-L/100km"]=235/df["city-mpg"]
df.head()


# In[14]:


df["highway-L/100km"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)
df.head()


# ## Data Standardization
# <b>Why normalization?</b>
# <p>Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include 
# <ol>
#     <li>scaling the variable so the variable average is 0</li>
#     <li>scaling the variable so the variance is 1</li> 
#     <li>scaling the variable so the variable values range from 0 to 1</li>
# </ol>
# </p>
# 
# <b>Example</b>
# <p>To demonstrate normalization, say you want to scale the columns "length", "width" and "height".</p>
# <p><b>Target:</b> normalize those variables so their value ranges from 0 to 1</p>
# <p><b>Approach:</b> replace the original value by (original value)/(maximum value)</p>

# In[16]:


df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/ df["width"].max()
df["height"] = df["height"]/df["height"].max()


# In[17]:


df[["length", "width","height"]].head()


# ## Binning
# <b>Why binning?</b>
# <p>
#     Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.
# </p>
# 
# <b>Example: </b>
# <p>In your data set, "horsepower" is a real valued variable ranging from 48 to 288 and it has 59 unique values. What if you only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? You can rearrange them into three ‘bins' to simplify analysis.</p>
# 
# <p>Use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins.</p>

#  Convert data to correct format:
# 

# In[18]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# Plot the histogram of horsepower to see the distribution of horsepower.
# 

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>Find 3 bins of equal size bandwidth by using Numpy's <code>linspace(start_value, end_value, numbers_generated</code> function.</p>
# <p>Since you want to include the minimum value of horsepower, set start_value = min(df["horsepower"]).</p>
# <p>Since you want to include the maximum value of horsepower, set end_value = max(df["horsepower"]).</p>
# <p>Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4.</p>

# Build a bin array with a minimum value to a maximum value by using the bandwidth calculated above. The values will determine when one bin ends and another begins.
# 

# In[20]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]),4)
bins


# In[21]:


group_names = ['Low', 'Medium', 'High']


# Apply the function "cut" to determine what each value of `df['horsepower']` belongs to. 
# 

# In[24]:


df["horsepower-binned"] = pd.cut(df["horsepower"],bins,labels=group_names, include_lowest = True)
df[["horsepower","horsepower-binned"]].head(5)


# See the number of vehicles in each bin:

# In[25]:


df["horsepower-binned"].value_counts()


# Plot the distribution of each bin:
# 

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>
#     Look at the data frame above carefully. You will find that the last column provides the bins for "horsepower" based on 3 categories ("Low", "Medium" and "High"). 
# </p>
# <p>
#     You successfully narrowed down the intervals from 59 to 3!
# </p>

# ## Indicator Variable
# <b>What is an indicator variable?</b>
# <p>
#     An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning. 
# </p>
# 
# <b>Why use indicator variables?</b>
# <p>
#     You use indicator variables so you can use categorical variables for regression analysis in the later modules.
# </p>
# <b>Example</b>
# <p>
#     The column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, you can convert "fuel-type" to indicator variables.
# </p>
# 
# <p>
#     Use the Panda method 'get_dummies' to assign numerical values to different categories of fuel type. 
# </p>

# In[28]:


df.columns


# Get the indicator variables and assign it to data frame "dummy_variable_1":
# 

# In[30]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[31]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In the data frame, column 'fuel-type' now has values for 'gas' and 'diesel' as 0s and 1s.
# 

# In[32]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[33]:


df.head()


# In[34]:


df.to_csv('clean_auto.csv')


# In[ ]:




