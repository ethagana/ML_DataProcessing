
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

#Reshape Y into 2D array
Y = Y.reshape(len(Y),1)
print("\nConvert Y to 2D array\n\n")
print(Y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)
#X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("\nRunning Feature Scaling\n\n")
print(X)
print("\n")
print(Y)

#Training the SVR model on complete dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predict a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)))

#Visualising the SVR results
#plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(Y), color = 'red')
#plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
#plt.title('Truth or Bluff (SVR Regression)')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.show()

#Visualising the SVR with smoother curve
X_grid = np.arange(min(sc_x.inverse_transform(X)),max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(X), sc_y.transform(Y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()