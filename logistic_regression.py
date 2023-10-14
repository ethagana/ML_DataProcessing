# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Training the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
print("\nLogistical Regression Training Output\n")
print(classifier.fit(X_train, y_train))

# Predict a new result
print("\nLogistical Regression Predict at Age 30\n")
print(classifier.predict(sc.transform([[30,8700]])))

#Predicting the Test set Results
y_pred = classifier.predict(X_test)
#np.set_printoptions(precision=2)
print("\nPredicted Results\n")
#print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#((,y_test.reshape(len(y_test,1)))),1
#print("\n\nRunning Prediction")
#print(regressor.predict([[1,0,0,160000,130000,300000]]))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,y_pred))
print("\nAccuracy Score\n")
print(accuracy_score(y_test,y_pred))
