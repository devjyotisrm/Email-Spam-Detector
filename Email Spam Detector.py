#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[18]:


data = pd.read_csv('C:/Users/DELL/Desktop/emails.csv')


# In[19]:


data_clean = data.loc[:,['text','spam']]


# In[20]:


data_clean


# In[21]:


count=data_clean['spam'].value_counts()
count.to_frame()


# In[22]:


data_clean = data_clean[(data_clean['spam']=='0')|(data_clean['spam']=='1')]


# In[23]:


data_clean


# In[24]:


X = data_clean.loc[:,'text']
y = data_clean.loc[:,'spam']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)


# In[25]:


count_vect = CountVectorizer()
xtrain = count_vect.fit_transform(X_train) 
xtest=count_vect.transform(X_test) 


# In[26]:


model = MultinomialNB().fit(xtrain,y_train)
y_pred= model.predict(xtest) 


# In[27]:


y_pred


# In[35]:


print(accuracy_score(y_test,y_pred))


# In[34]:


print(confusion_matrix(y_test,y_pred))


# In[33]:


print(classification_report(y_test,y_pred))


# In[ ]:




