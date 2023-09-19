# Importing libraries
from sklearn.tree import DecisionTreeRegressor
import numpy
import pandas
import matplotlib.pyplot as plt

# importing dataset
dataset = pandas.read_csv(r"D:\Machine Learning\Machine Learning\04 Decision Tree Regression\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
# print(X)
# print(Y)

# Splitting the set to training and test set
# Not required as there very less observations

# Feature scaling is also not required

# Fitting Decision Tree regression to the dataset
regressor = DecisionTreeRegressor(random_state=0, )
regressor.fit(X, Y)

# Predicting a new result
Y_predicted = regressor.predict([[6.5]])
print(Y_predicted)

## Visualising the Decision Tree results
# plt.scatter(X, Y, color="red")
# plt.plot(X, regressor.predict(X), color="blue")
# plt.title("Position vs Salary (Decision Tree Reg.)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# Visualizing Decision Tree regression results in Higher Resolution
X_grid = numpy.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Position vs Salary (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
