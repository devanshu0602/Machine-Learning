# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv(r"D:\Machine Learning\Machine Learning\06 Random Forest Regression\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# fit the random forest regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, Y)

# Predicting new results
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualizin the regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Position vs Salary (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()