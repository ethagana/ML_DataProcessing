#Import the Liblaries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Import the dataset
dataset = pd.read_csv("50_Startups.csv");
X = dataset.iloc[:, :-1].values;
Y = dataset.iloc[:,-1].values;
print("Imported Data Set\n")
print(X)
print("\n")
print(Y)


#Encoding the independent Variable
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough');
X = np.array(ct.fit_transform(X));
print('\n\nEncoding Independent Variables\n\n')
print(X); 

#Splitting the dataset into training and test set respectively
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1);

#Training the Multi Linear Regression model on the Training data
#NB You don't have to worry about the dummy variables
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test set Results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print("\nPredicted Results\n")
#print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))
#((,y_test.reshape(len(y_test,1)))),1
print("\n\nRunning Prediction")
print(regressor.predict([[1,0,0,160000,130000,300000]]))