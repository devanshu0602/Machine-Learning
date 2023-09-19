# THEORY
# Equation can be given as:
# y = b0 + b1*x1 + ... + bnxn
# where, y = dependent variable
#        x1 = independent variable
#        b0,...,bn = constants 


# -------- Data Preprocessing --------
# Importing required libraries
import pandas
import numpy
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pandas.read_csv(r"D:\Machine Learning\Machine Learning\02 Multiple Linear Regression - Find Best Investment\50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values    # profit column
#print(X)
#print(Y)

# Encoding categorical data (independent variable)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
onehotencoder = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder="passthrough")
X = onehotencoder.fit_transform(X)
#print(X)

# Avoiding dummy variable trap - by getting rid of one dummy variable (1st column)
X = X[:, 1:]
#print(X)

# Splitting dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(Y_test)
# --------------------------------


# -------- Regression Model --------
# Fitting Multiple linear regression model to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting Test set results
Y_predicted = regressor.predict(X_test)
# print(Y_predicted)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# the next line adds a column of 1s to X -> to be used as b0 in the line equation
# X = numpy.append(arr=X, values=numpy.ones((50, 1)).astype(int), axis=1)
# in order to add the column at the beginning, we invert arr & values, i.e., we add X to the column
X = numpy.append(arr=numpy.ones((50, 1)).astype(int), values=X, axis=1)
#print(X)


# Creating an optimal matrix of features - independent var. that have high impact
# OLS = Ordinary Least Squares
X_optimal = numpy.array(X[:, [0,1,2,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary() # tells us that the highest P-value is of x2 (index 2/column 3) and hence, we remove column 3

X_optimal = numpy.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary() # tells us that the highest P-value is of x1 (index 1/column 2) and hence, we remove column 2

X_optimal = numpy.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary() # tells us that the highest P-value is of x2 (index 2/column 3) and hence, we remove column 3

X_optimal = numpy.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary() # tells us that the highest P-value is of x2 (index 2/column 3) and hence, we remove column 3

X_optimal = numpy.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
print(regressor_OLS.summary()) # tells us that the highest impact is caused by only one independent variable -> R&D spend

# From the summary table, we get the coefficients of x1 and also get b0
# and hence, we now have the equation of the Multiple Linear Regression


# ALTERNATIVE for Backward Elimination :
# def backwardElimination(x, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar & gt; sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#     regressor_OLS.summary()
#     return x


# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)