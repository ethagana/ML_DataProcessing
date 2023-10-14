
#Import the Liblaries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Import the dataset
dataset = pd.read_csv("Salary_Data.csv");
X = dataset.iloc[:, :-1].values;
Y = dataset.iloc[:,-1].values;

#Splitting the dataset into training and test set respectively
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1);

#Training the simple linear regressions model on the training set
from sklearn.linear_model import LinearRegression;
regressor = LinearRegression();
regressor.fit(X_train,Y_train);

#Prediciting the Test set results
y_pred = regressor.predict(X_test);

#Visualise Training set results
#plt.scatter(X_train, Y_train, color = 'red');
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary Vs Experience (Training Set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show();

#Visualise Test set results
plt.scatter(X_test, Y_test, color = 'red');
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show();

#Predict salary of employee who has been with the firm for 12 years
print("Predict salary of employee with 12 years")
print(regressor.predict([[12]]));

#Get final linear regression equation with the values of the coefficients
print("\n\nPrint Coefficients\n\n")
print(regressor.coef_);
print("\n\n")
print(regressor.intercept_)