## Import libraries
import numpy
import pandas
import matplotlib.pyplot as plt

## Import dataset
dataset = pandas.read_csv(r"D:\Machine Learning\Machine Learning\03 Polynomial Regression\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
# print(X)
# print(Y)

## Splitting the set to training and test set
# Not required as there very less observations

## Feature scaling is also not required

## Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
Lin_Reg = LinearRegression()
Lin_Reg.fit(X, Y)

## Fitting polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
Poly_Reg = PolynomialFeatures(degree = 4)
X_poly = Poly_Reg.fit_transform(X)
Lin_Reg_2 = LinearRegression()
Lin_Reg_2.fit(X_poly, Y)

## Visualizing Linear regression results
plt.scatter(X, Y, color='red')
plt.plot(X, Lin_Reg.predict(X), color='blue')
plt.title('Position vs Salary (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Visualizing Polynomial regression results -> X_grid is used for Higher Resolution
# X_grid = numpy.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
# plt.plot(X_grid, Lin_Reg_2.predict(Poly_Reg.fit_transform(X_grid)), color = 'blue')
plt.plot(X, Lin_Reg_2.predict(Poly_Reg.fit_transform(X)), color='blue')
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result with Linear Regression
print("Salary prediction using Linear Reg = ", Lin_Reg.predict([[6.5]]))

## Predicting a new result with Polynomial Regression
print("Salary Prediction using Polynomial Reg = ", Lin_Reg_2.predict(Poly_Reg.fit_transform([[6.5]])))