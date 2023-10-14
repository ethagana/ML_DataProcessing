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

#Training the Decision Tree Regression model on whole dataset
from sklearn.tree  import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
print("\nDecision Tree Training Output\n")
print(regressor.fit(X,Y))

#Predict a new result
regressor.predict([[6.5]])

#Visualize the Decision Tree Regression results with smoother curve

X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()