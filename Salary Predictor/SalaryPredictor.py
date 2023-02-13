# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing datasets
dataset = pd.read_csv(r"D:\Machine Learning\Machine Learning\Salary Predictor\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# splitting dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()      # simple linear regressor 
regressor.fit(X_train, Y_train) 

# predicting the test set results
Y_predicted = regressor.predict(X_test)     # predicted results

# visualizing the training set results
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualizing the test set results
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()