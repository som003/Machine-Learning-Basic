# Logistic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# standardisation is -1 to 1 and normilazation is 0 to 1.

# Load the diabetes dataset
diabetes = load_diabetes() # control click diabetes_data for seeing the decomposition 
# between data and target in sk learn in diabetes data and target are already defined 
# in 2 different csv files.

X, y = diabetes.data, diabetes.target

y_binary = (y > np.median(y)).astype(int)
# Convert the target variable to binary (1 for diabetes, 0 for no diabetes
# here the y data is distributed and the medain is found, number above median are 1 
# and below median are 0.


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
'''
random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

for more information crtl + click random_state
'''

# Standardize features
scaler = StandardScaler()
"""Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.
    
for more info ctrl + click StandardScalar()
"""
X_train = scaler.fit_transform(X_train)
"""
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
for more info ctrl + click fit_transform()  

The fit() method calculates the required parameters, and the transform() method 
applies the calculated parameters to standardize the data. For example, the 
StandardScaler() calculates the mean and standard deviation, and scales and centers 
the data to have a mean of 0 and a standard deviation of 1. 
The fit_transform() method saves time and effort by combining both fit() and 
transform() calls
"""

X_test = scaler.transform(X_test)
"""Perform standardization by centering and scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


class_names=[0,1] # name of classes
fig, ax = plt.subplots()
plt.show()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.show()

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='gax.xaxis.set_label_position("top")')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
# Text(0.5,257.44,'Predicted label')


