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

# Training the Naive Bayse model on the training set
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predict a new result
print("\nNaive Bayes Purchase Predict at Age 30\n")
print(classifier.predict(sc.transform([[30,87000]])))

