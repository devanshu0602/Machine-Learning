"""
THEORY
Equation can be given as:
y = b0 + b1*x1
where, y = dependent variable
       x1 = independent variable
       b0, b1 = constant 
"""

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv(r"D:\Machine Learning\Machine Learning\01 Simple Linear Regression - Salary Predictor\Salary Predictor\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting simple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()      # simple linear regressor 
regressor.fit(X_train, Y_train) 

# Predicting the Test set results
Y_predicted = regressor.predict(X_test)     # predicted results

# VISUALIZATION
# Visualizing the Training set results
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
# Visualizing the Test set results
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()