#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[2]:


# read the data using pandas dataframe
clicks_df = pd.read_csv('clicks_dataset.csv', encoding='ISO-8859-1')


# In[5]:


# Show the data head!
clicks_df.head()


# In[6]:


# Show the data head!
clicks_df.tail()


# **PRACTICE OPPORTUNITY #1 [OPTIONAL]:**
# - **Display the first and last 10 rows of the dataframe** 
# - **Calculate the average salary**

# In[7]:


clicks_df.mean()


# # TASK #3: EXPLORE DATASET

# In[9]:


click=clicks_df[clicks_df['Clicked']==1]
no_click=clicks_df[clicks_df['Clicked']==0]


# In[10]:


print("Total =", len(clicks_df))

print("Number of customers who clicked on Ad =", len(click))
print("Percentage Clicked =", 1.*len(click)/len(clicks_df)*100.0, "%")
 
print("Did not Click =", len(no_click))
print("Percentage who did not Click =", 1.*len(no_click)/len(clicks_df)*100.0, "%")
 
        


# In[11]:


clicks_df.describe()


# # TASK #4: PERFORM DATA VISUALIZATION

# In[16]:


sns.scatterplot(data=clicks_df,
                x='Time Spent on Site',
                y='Salary',
                hue='Clicked')


# In[17]:


plt.figure(figsize=(5, 5))

sns.boxplot(x='Clicked',y='Salary',data=clicks_df)


# In[18]:


clicks_df['Salary'].hist(bins=40)


# **PRACTICE OPPORTUNITY #2 [OPTIONAL]:**
# - **Plot the histogram of "Time Spent on Site", use 30 bins.**
# - **Plot the boxplot showing "clicked" vs. "Time Spent on Site".**
# 

# In[19]:


clicks_df['Time Spent on Site'].hist(bins=30)


# # TASK #5: PREPARE THE DATA FOR TRAINING

# In[20]:


clicks_df


# In[21]:


#Let's drop the emails, country and names (we can make use of the country later!)
clicks_df.drop(['Names', 'emails', 'Country'],axis=1,inplace=True)


# In[22]:


clicks_df


# In[23]:


#Let's drop the target coloumn before we do train test split
X = clicks_df.drop('Clicked',axis=1).values
y = clicks_df['Clicked'].values


# In[24]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# # TASK #6: PERFORM MODEL TRAINING

# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:




X_train
# In[26]:


y_train


# In[27]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# # TASK #7: TEST TRAINED MODEL 

# In[28]:


y_predict_train=model.predict(X_train)


# In[29]:


y_train


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True,fmt='d')


# In[32]:


y_predict_test = model.predict(X_test)
y_predict_test


# In[33]:


cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True, fmt="d")


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_test))


# # TASK #8: VISUALIZE TRAINING AND TESTING DATASETS

# In[42]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# Create a meshgrid ranging from the minimum to maximum value for both features

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


# In[43]:


y_train.shape


# In[44]:


X_train.shape


# In[45]:


X1.shape


# In[46]:


# plot the boundary using the trained classifier
# Run the classifier to predict the outcome on all pixels with resolution of 0.01
# Colouring the pixels with 0 or 1
# If classified as 0 it will be magenta, and if it is classified as 1 it will be shown in blue 
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


# In[47]:


# plot all the actual training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
    
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[48]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[49]:


# Visualising the testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# 

# In[ ]:





# # GREAT JOB!

# **PRACTICE OPPORTUNITY #1 SOLUTION:**
# - **Display the first and last 10 rows of the dataframe** 
# - **Calculate the average salary**

# In[35]:


clicks_df['Salary'].mean()


# **PRACTICE OPPORTUNITY #2 SOLUTION:**
# - **Plot the histogram of "Time Spent on Site", use 30 bins.**
# - **Plot the boxplot showing "clicked" vs. "Time Spent on Site".**
# 

# In[36]:


plt.figure(figsize=(5, 5))
sns.boxplot(x='Clicked', y='Time Spent on Site',data=clicks_df)


# In[37]:


clicks_df['Time Spent on Site'].hist(bins = 20)


# **PRACTICE OPPORTUNITY #3 SOLUTION:**
# - **Retrain an XG-Boost Model instead and assess its performance. External Research is required.** 

# In[38]:


# Fitting XG-Boost to the Training set
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
model.score(X_test, y_test)

