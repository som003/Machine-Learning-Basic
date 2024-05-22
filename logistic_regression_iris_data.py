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

# we have not standardise  the target variable 'y' because it is categorical and cannot be standardised

# 3 types of classification binary, ordinal and multi. Multi-class classification has more than three classes.

# LINEAR REGRESSION

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

#predicting on test set
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy of logistic regression model : ',accuracy)
y_pred_train = lr.predict(X_train)
print("Accuracy of train : ",accuracy_score(y_train,y_pred_train))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('\n\nClassification Report:\n',classification_report(y_test,y_pred))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,y_pred))
cnf = confusion_matrix(y_test,y_pred)

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()