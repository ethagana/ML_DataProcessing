#Import the Liblaries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Import the dataset
dataset = pd.read_csv("Data.csv");
X = dataset.iloc[:, :-1].values;
Y = dataset.iloc[:,-1].values;

print(X);
print(Y);

print('\n\nCleaning Data...\n\n');

#Taking care of missing data - replace with average/median/popular of all values in the chart
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median');
imputer.fit(X[:, 1:3]);
X[:, 1:3] = imputer.transform(X[:, 1:3]);
print(X);

#Encoding categorical data one hot encoding

#Encoding the independent Variable
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough');
X = np.array(ct.fit_transform(X));
print('\n\nEncoding Independent Variables\n\n')
print(X);

#Encoding the Dependent Variable - Label Encoding
from sklearn.preprocessing import LabelEncoder;
le = LabelEncoder();
Y = le.fit_transform(Y)
print('\n\nEncoding Y variables\n\n')
print(Y);

#Splitting the dataset into training and test set respectively
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1);
print('\n\nTraining and Test Sets\n\n')
print(X_train);
print('\n\n');
print(X_test);
print('\n\n');
print(Y_train);
print('\n\n');
print(Y_test);
print('\n\n');


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler();
X_train[:,3:] = sc.fit_transform(X_train[:, 3:]);
X_test[:,3:] = sc.transform(X_test[:, 3:]);

print("\n\nFeature Scaling of Training and Test set")
print(X_train);
print("\n\n");
print(X_test);
print("\n\n");