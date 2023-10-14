# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values #Matrix of features
#y = dataset.iloc[:, -1].values
print(X)
#print(y)

#Using the dendrogram to find the optimum number of clusters
#import scipy.cluster.hierarchy as sch
#dendrogram = sch.dendrogram(sch.linkage(X, method= 'ward'))
#plt.title('Dendrogram')
#plt.xlabel('Customers')
#plt.ylabel('Euclidean Distances')
#plt.show()

#Training the Model on the Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)
print(y_hc)

#Visualise the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'Red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'Blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'Green', label = 'Cluster 3')
#plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'Cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'Magenta', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],s = 300, c = 'Yellow', label = 'Centroids')
#plt.plot(range(1,11), wcss)
plt.title('Cluster of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()