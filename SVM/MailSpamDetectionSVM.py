# -*- coding: utf-8 -*-
"""
Definition : Mail Spam Detection Using SVM

Created on Thu May 31 15:00:33 2018

@author: Gaurav Shimpi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Reading Data form csv File...
data = pd.read_csv("MailSpamDetectionSVM.csv")
#Convert Data into Dataframe...
data = pd.DataFrame(data)

#Splitting Data for Training and Testing...
train, test = train_test_split(data, test_size=0.2)

# The columns that we will be making predictions with.
x_columns = ["Number of recipients", "size in kb"]
# The column that we want to predict.
y_column = ["spam/not spam"]

# Create the knn model.
# Look at the five closest neighbors.
svclassifier = SVC(kernel='linear')

# Fit the model on the training data.
svclassifier.fit(train[x_columns], train[y_column])

#Get Data From User
print("...Welcome to K Nearest Neighbour Example...")
rec = int(input("Enter Number of Recipients:"))
size= int(input("Enter Size in KB:"))
user_test_data = np.array([rec,size]).reshape(1,2)

predicted_value = int(svclassifier.predict(user_test_data).round())
print("Prediction value is:", predicted_value,"So, your ")
if(predicted_value == 0):
    print("Mail is not Spam...")
else:
    print("Mail is Spam...")
predictions_for_accuracy = svclassifier.predict(test[x_columns])
acc= accuracy_score(test[y_column], predictions_for_accuracy.round())
print("Accuracy",(acc*100).round(),"%")

"""Retraining The Model for New Data"""
choice = input("DO you want to add data into file? y/n")
if(choice == 'y'):
    l = len(data)
    data.loc[l]=[l,rec,size,predicted_value]
    data = pd.DataFrame({'Number of recipients':data['Number of recipients'],'size in kb':data['size in kb'],'spam/not spam':data['spam/not spam']})
    data.to_csv("MailSpamDetectionSVM.csv")
    print("Data is added!!!")
else:
    print("Thank You !!!")
    
    
'''
Created on Sat Nov 24 12:04:55 2018

@author: Gaurav Shimpi

from sklearn.metrics import classification_report, confusion_matrix
print("\n",confusion_matrix(y_test,y_pred))
print("\n",classification_report(y_test,y_pred))

'''