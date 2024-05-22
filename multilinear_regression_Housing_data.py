# predict the price of house using linear regression.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv(
    "C:\\Users\\Om Shah\\Desktop\\python class\\ML - CLASS SEM 6\\Housing.csv")
data.info()
data.describe()

y = data[['price']]
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

# reg coeff and intercept are:
print(f'coefficient: {reg.coef_}')
print(f'Intercept: {reg.intercept_}')
