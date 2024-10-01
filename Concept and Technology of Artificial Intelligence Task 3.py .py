#!/usr/bin/env python
# coding: utf-8

# # Task 3 Logistic Regression, Neural Networls and Gaussian Naive Bayes

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


data=pd.read_csv(r'nba_rookie_data.csv')


# In[3]:


data.head() 


# In[4]:


data.shape # this tells us the number of rows and cloumns in the dataset


# #### Here the target variable is "TARGET_5Yrs"  

# In[5]:


# In this analysis we will apply Logistic Regreesion, Gaussiian Naive Bayes Algorithm and Multi Layer Perceptron to predict
# the target variable and later on we will compare these techniques.


# In[6]:


# Firstly we will apply Logistic regression (Simple and multiple)


# In[7]:


# let us check for null values in the variables
data.isna().sum()


# In[8]:


# we can see that '3 Point Percent' has 11 null values therefore we will impute these null values with mode.
m=data['3 Point Percent'].mode()
m


# In[9]:


data['3 Point Percent']=data['3 Point Percent'].fillna(0)
data['3 Point Percent'].isna().sum()


# In[10]:


#Transforming data before analysis
for i in data.columns:
    data[i]=LabelEncoder().fit_transform(data[i])


# In[11]:


data.head()


# ### Logistic Regression

# In[12]:


# Applying simple logostic regression
X=data[['Points Per Game']]
y=data['TARGET_5Yrs']


# In[13]:


#Splitting the data 
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)


# In[14]:


# fitting the model
logre=LogisticRegression()
logre.fit(X_train, y_train)


# In[15]:


# looking at the prediction
fig1, ax1 =plt.subplots()
ax1.scatter(X_test,y_test, color='blue')
ax1.scatter(X_test, logre.predict(X_test), color='red', marker='*')
ax1.scatter(X_test, logre.predict_proba(X_test)[:,1], color='g', marker='o')


# In[16]:


print('number of misslabeled points out of a total of %d points:%d' %(X_test.shape[0],(y_test!=logre.predict(X_test)).sum()))


# In[17]:


print('Our accuracy is: %.2f' %logre.score(X_test, y_test) )


# In[18]:


# Including more variables to see if we can increase the accuracy


# In[19]:


X=data[['Games Played', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[20]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
logre=LogisticRegression()
logre.fit(X_train, y_train)


# In[21]:


print('number of misslabeled points out of a total of %d points:%d' %(X_test.shape[0],(y_test!=logre.predict(X_test)).sum()))
print('Our accuracy is: %.2f' %logre.score(X_test, y_test) )


# In[22]:


# Accuracy rate is improved by using two independent variables.
# Now we will build a regression model based on linear correlation of variables with the target variables
c=data.corr().abs()
s=c.unstack()
print(s['TARGET_5Yrs'].sort_values(kind='quicksort', ascending= False))


# ##### the values of correlation coefficients are low, hence considering variables with correlation coeffient > 0.3 to build a new model 

# In[23]:


X=data[['Games Played', 'Points Per Game', 'Field Goals Made', 'Field Goals Made', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[24]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
logre=LogisticRegression()
logre.fit(X_train, y_train)


# In[25]:


print('number of misslabeled points out of a total of %d points:%d' %(X_test.shape[0],(y_test!=logre.predict(X_test)).sum()))
print('Our accuracy is: %.2f' %logre.score(X_test, y_test) )


# In[26]:


# unfortunately the accuracy score is not improved 


# ### Gaussian Naive Bayes.

# In[26]:


# Let us build the model by considering 1 and 2 independent variables at a time.
X=data[['Points Per Game']]
y=data['TARGET_5Yrs']


# In[27]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
gnb=GaussianNB()
gnb.fit(X_train, y_train)


# In[28]:


print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!=gnb.predict(X_test)).sum()))
print('Our accuracy is: %.2f' %gnb.score(X_test, y_test) )


# In[29]:


#plotting Probabilities
fig1, ax1 =plt.subplots()
ax1.scatter(X_test, y_test, color='b')
ax1.scatter(X_test, gnb.predict(X_test), color='r', marker='*')
ax1.scatter(X_test, gnb.predict_proba(X_test)[:,1], color='g', marker='o')
plt.show()


# In[30]:


# let us now try using two varibales as independent features and see if accuracy improves


# In[31]:


X=data[['Games Played', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[32]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
gnb=GaussianNB()
gnb.fit(X_train, y_train)


# In[33]:


print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!=gnb.predict(X_test)).sum()))
print('Our accuracy is: %.2f' %gnb.score(X_test, y_test) )


# In[34]:


# The accuracy value is increased to 0.72. Let us try including more features to improve accuracy based on correlation criteria


# In[35]:


X=data[['Games Played', 'Points Per Game', 'Field Goals Made', 'Field Goals Made', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[36]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
gnb=GaussianNB()
gnb.fit(X_train, y_train)


# In[37]:


print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!=gnb.predict(X_test)).sum()))
print('Our accuracy is: %.2f' %gnb.score(X_test, y_test) )


# In[38]:


# No the accuracy value is decreased by adding more variables


# ### MLP

# In[39]:


# for this technique also, starting with one variable, then using 2 variables and then inccluding variables as per correlation coefficient.
X=data[['Points Per Game']]
y=data['TARGET_5Yrs']


# In[40]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
mlp=MLPClassifier(hidden_layer_sizes=(20, 30), activation='logistic', random_state=0, max_iter=2000)
mlp.fit(X_train, y_train)


# In[41]:


print('Our Accuracy is %.2f' %mlp.score(X_test,y_test))
print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!= mlp.predict(X_test)).sum()))


# In[42]:


# using two independent features.
X=data[['Games Played', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[43]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
mlp=MLPClassifier(hidden_layer_sizes=(20, 15), activation='logistic', random_state=0, max_iter=2000)
mlp.fit(X_train, y_train)


# In[44]:


print('Our Accuracy is %.2f' %mlp.score(X_test,y_test))
print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!= mlp.predict(X_test)).sum()))


# In[45]:


# Here the accuracy imporved to 0.70. 


# In[46]:


# including more variables (with correlation coefficient >0.3)


# In[47]:


X=data[['Games Played', 'Points Per Game', 'Field Goals Made', 'Field Goals Made', 'Minutes Played']]
y=data['TARGET_5Yrs']


# In[48]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)
mlp=MLPClassifier(hidden_layer_sizes=(20, 20), activation='logistic', random_state=0, max_iter=2000)
mlp.fit(X_train, y_train)


# In[49]:


print('Our Accuracy is %.2f' %mlp.score(X_test,y_test))
print('Number of mislabeled points out of %d are: %d' %(X_test.shape[0], (y_test!= mlp.predict(X_test)).sum()))


# In[50]:


# here we can see that the accuracy score is decreased by including more variables.

