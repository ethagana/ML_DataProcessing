#Import the Liblaries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Import the dataset
dataset = pd.read_csv("Position_Salaries.csv");
X = dataset.iloc[:, 1:-1].values;
Y = dataset.iloc[:,-1].values;
print("Imported Data Set\n")
print(X)
print("\n")
print(Y)

#Train the Linear Regression Model on whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Training the Polynomial Regressions Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,Y)

#Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#plt.show()

#Predicting a new result with Linear Regression
print("\nSalary Prediction using Linear Regression")
print(lin_reg.predict([[6.5]]))

#Predicted salary on Polynomial Regression 
print("\nSalary Prediction using Polynomial Regression")
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))