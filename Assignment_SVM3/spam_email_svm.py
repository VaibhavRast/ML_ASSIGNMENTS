#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


spammail=pd.read_csv('spambase.csv')


# In[15]:


spammail.head()


# In[16]:


spammail.shape


# In[3]:


X=spammail.drop('spam',axis=1)


# In[4]:


y=spammail['spam']


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[6]:


#Feature scaling-standardizing features by removing mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
X_Train=scaler_x.fit_transform(X_train)
X_Test=scaler_x.transform(X_test)


# In[17]:


tuned_parameters_quad = [{'kernel':['poly'],'degree':[2],'C':[1,10,100,1000,10000,50000,100000]}]
tuned_parameters_linear=[{'kernel':['linear'],'C':[1,2]}]
tuned_parameters_rbf=[{'kernel':['rbf'],'C':[1,10,100,1000,10000,100000],'gamma':['scale','auto']}]


# In[8]:


#SVM model to predict if a mail is spam or non spam
#In order to vary regulation parameter C and decide an optimal value, we are using an exhaustive grid search
#C has been given the values of 1,10,100,1000 and 10000

def svmmodel(tuned_parameters):
  from sklearn.svm import SVC
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import confusion_matrix
  svclassifier = GridSearchCV(SVC(), param_grid=tuned_parameters, scoring='accuracy',verbose=10,n_jobs=-1)
  svclassifier.fit(X_train, y_train)
  print('Scores:')
  means = svclassifier.cv_results_['mean_test_score']
  stds = svclassifier.cv_results_['std_test_score']
  for mean, std, params in zip(means, stds, svclassifier.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
  print('Best score:')
  print(svclassifier.best_params_)
  y_true_test, y_predtest = y_test, svclassifier.predict(X_test)
  y_true_train, y_predtrain = y_train, svclassifier.predict(X_train)
  cfmatrixtrain=confusion_matrix(y_true_train,y_predtrain)
  cfmatrixtest=confusion_matrix(y_true_test,y_predtest)
  cfmetrics(cfmatrixtrain,cfmatrixtest)


# In[9]:


def cfmetrics(cfmatrixtrain,cfmatrixtest):  
  print('confusion matrix for training data:')
  print(cfmatrixtrain)
  TN=cfmatrixtrain[0][0]
  FN=cfmatrixtrain[1][0]
  TP=cfmatrixtrain[1][1]
  FP=cfmatrixtrain[0][1]
  accuracy_train=(TN+TP)/(TN+TP+FN+FP)
  precision_train=(TP)/(TP+FP)
  recall_train=TP/(TP+FN)
  print('Training accuracy')
  print(accuracy_train)
  print('Training precision')
  print(precision_train)
  print('Training recall')
  print(recall_train)
  print('confusion matrix for test data:')
  print(cfmatrixtest)
  TN=cfmatrixtest[0][0]
  FN=cfmatrixtest[1][0]
  TP=cfmatrixtest[1][1]
  FP=cfmatrixtest[0][1]
  accuracy_test=(TN+TP)/(TN+TP+FN+FP)
  precision_test=(TP)/(TP+FP)
  recall_test=TP/(TP+FN)
  print('Test accuracy')
  print(accuracy_test)
  print('Test precision')
  print(precision_test)
  print('Test recall')
  print(recall_test)


# In[10]:


#Linear kernel function for SVM
svmmodel(tuned_parameters_linear)


# In[18]:


#Quadratic kernel function for SVM
svmmodel(tuned_parameters_quad)


# In[12]:


#RBF kernel function for SVM
svmmodel(tuned_parameters_rbf)


# In[13]:


#RBF kernel function for SVM
svmmodel(tuned_parameters_rbf)


# In[14]:


tuned_parameters_linear=[{'kernel':['linear'],'C':[0.1,1]}]
svmmodel(tuned_parameters_linear)

