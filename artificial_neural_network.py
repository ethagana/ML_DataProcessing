# Importing the libraries
import numpy as np
import tensorflow as tf
import pandas as pd

print(tf.__version__)

#Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

#exit(0)

# Encoding the Dependent Variable - Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

#exit(0)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#exit(0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train)
#print(X_test)

# Building the ANN

#Initializing the ANN
ann = tf.keras.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer - use Sigmoid function
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training the ANN

#Compiling the ANN - categorical_crossentropy for non binary loss
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Predict if client will leave bank - Homework
print('\nPredicting if a client will leave the bank\n')
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#Predict the test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Make the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,y_pred))
print("\nAccuracy Score\n")
print(accuracy_score(y_test,y_pred))
