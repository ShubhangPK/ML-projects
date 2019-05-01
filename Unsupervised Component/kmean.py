import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
xlab = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans
ylab = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(xlab)
    ylab.append(kmeans.inertia_)
plt.plot(range(1, 11), ylab)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('lab')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(xlab)


plt.scatter(xlab[y_kmeans==0,0],xlab[y_kmeans==0,1],s=100,c='blue',label='Cluster 1')
plt.scatter(xlab[y_kmeans==1,0],xlab[y_kmeans==1,1],s=100,c='green',label='Cluster 2')
plt.scatter(xlab[y_kmeans==2,0],xlab[y_kmeans==2,1],s=100,c='grey',label='Cluster 3')
plt.scatter(xlab[y_kmeans==3,0],xlab[y_kmeans==3,1],s=100,c='red',label='Cluster 4')
plt.scatter(xlab[y_kmeans==4,0],xlab[y_kmeans==4,1],s=100,c='orange',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label= 'Centroids')
plt.title('Customer Clusters')
plt.xlabel('Annual Income in Dollars')
plt.ylabel('SpendingScore(1-100)')
plt.legend()
plt.show()
