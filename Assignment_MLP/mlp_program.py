# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:42:45 2020

@author: Vaibhav Rastogi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# i) Reading Data
dataset = pd.read_csv('Insuarance_Policy_Renewal.csv')

# Data Preprocessing Stage

#Categorical Data
dataset['Gender']=dataset['Gender'].map(
        {
            'Female':1,
            'Male':0
        }
    )


dataset['RenewsPolicy']=dataset['RenewsPolicy'].map(
        {
            'Yes':1,
            'No':0
        }
    )

dataset=dataset.drop(['Sno','CustomerId'],axis=1)


X = dataset.iloc[:,0:10].values
y = dataset.iloc[:, 10].values

#Dealing with MultiCategory Data Location 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_location = LabelEncoder()
X[:, 1] = labelencoder_location.fit_transform(X[:, 1])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Drop dummy variable to avoid dummy trap
X=X[:,1:]


# ii)Splitting Data int training and test set and setting random sate to 0
from sklearn.model_selection import train_test_split
X_trainSet, X_testSet, y_trainSet, y_testSet = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Applying Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainSet = sc.fit_transform(X_trainSet)
X_testSet = sc.transform(X_testSet)




#Importing Keras
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense


# Creating Perceptron

#Initialising the artificial neural network
model=Sequential()

#Adding input layer and one hidden layer

# iii)Initialising Weights to be uniform and iv) Defining activation function as rectifier
model.add(Dense(output_dim=8,init='uniform',activation='relu',input_dim=11))

#Adding output layer

# iv) Activation Function is sigmoid to get prbability as output
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Compiling the neural network
# iii)Defining learning rate=0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
print(K.eval(model.optimizer.lr))
#Fit the model to dataset
# iii) Define epochs=200 and v) Training the model
model.fit(X_trainSet,y_trainSet,batch_size=10,epochs=200)
print(K.eval(classifier.optimizer.lr))
# vi) Print learning rate,learnt weights and epochs
for layer in model.layers:
    learnt_wt=layer.get_weights()
print("Learnt Weights of the layers")
print(learnt_wt,end='\n')

print("HyperParameters")
print('No of epochs=200')

#vii)Prediction
#False- No(0), True-Yes(1)

#Training Set
y_predTrain = model.predict(X_trainSet)
y_predTrain=(y_predTrain>0.5)


#Test Set
y_predTest = model.predict(X_testSet)
y_predTest=(y_predTest>0.5)


#viii) Print Confusion Matrix and metrics
from sklearn.metrics import confusion_matrix

#Train Set
print('CONFUSION MATRIX FOR TRAINING SET')
cmTrainSet = confusion_matrix(y_trainSet, y_predTrain)
print(cmTrainSet,end='\n')
TN=cmTrainSet[0][0]
FN=cmTrainSet[1][0]
TP=cmTrainSet[1][1]
FP=cmTrainSet[0][1]
accuracy_train=(TN+TP)/(TN+TP+FN+FP)
precision_train=(TP)/(TP+FP)
recall=TP/(TP+FN)
print('METRICS FOR TRAINING SET:')
print('Accuracy:',accuracy_train,'Precision:',precision_train,'Recall:',recall)

#Test Set
cmTestSet = confusion_matrix(y_testSet, y_predTest)
print('CONFUSION MATRIX FOR TEST SET:')
print(cmTestSet,end='\n')
TN=cmTestSet[0][0]
FN=cmTestSet[1][0]
TP=cmTestSet[1][1]
FP=cmTestSet[0][1]
accuracy_test=(TN+TP)/(TN+TP+FN+FP)
precision_test=(TP)/(TP+FP)
recall_test=TP/(TP+FN)
print('METRICS FOR TRAINING SET:')
print('Accuracy:',accuracy_test,'Precision:',precision_test,'Recall:',recall_test)




 
