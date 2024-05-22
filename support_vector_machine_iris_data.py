import pandas as pd
import numpy as np

data = pd.read_csv("C:\\Users\\Om Shah\\Desktop\\python class\\ML - CLASS SEM 6\\Iris.csv")
#print(data.head)
#print(data.tail)

#print(data)

print(data.info())
print(data.describe())

from sklearn.model_selection import train_test_split
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
y = data[['Species']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy of SVM model : ',accuracy)
y_pred_train = clf.predict(X_train)
print("Accuracy of train : ",accuracy_score(y_train,y_pred_train))

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# print('\n\nClassification Report:\n',classification_report(y_test,y_pred))
# print('\nConfusion Matrix:\n',confusion_matrix(y_test,y_pred))