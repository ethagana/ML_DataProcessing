

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting=3)

#Cleaning the text
import re
import nltk 

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
corpus = []

for i in range(0, 1000):
    #Remove nonalphanumeric characters with a space
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = '  '.join(review)
    corpus.append(review)


#print(corpus)

#Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#print(X[0])

len(X[0])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Naive Bayse model on the training set
#from sklearn import naive_bayes
#classifier = naive_bayes.GaussianNB()
#classifier.fit(X_train,y_train)

# Training the Logistic Regression Model
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()
#print("\nLogistical Regression Training Output\n")
#print(classifier.fit(X_train, y_train))

#from sklearn import tree
#classifier = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)
#classifier.fit(X_train,y_train)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
#print("\nK Nearest Neighbour Training Output\n")
#print(classifier.fit(X_train, y_train))


# Training the Random forest classifier model on the training set
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy',random_state = 0)
#classifier.fit(X_train,y_train)

#Predicting the test results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1))))

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
#72 naive base
#78 logical regression
#73 decision tree classification
#66 K-nearest neighbour
#72 Random forest classifier



#Predicting if the Review is positive or negative

#for i in range(len(corpus) - 1):
   # new_review = corpus[i]
    #new_corpus = [new_review]
    #new_X_test = cv.transform(new_corpus).toarray()
    #new_y_pred = classifier.predict(new_X_test)
    #print(new_review + ' sentiment is ', new_y_pred)