import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('CLUSTERING/K-MEANS/Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11), wcss)
# plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
ykmeans = kmeans.fit_predict(x)

print(ykmeans)

plt.scatter(x[ykmeans==0,0], x[ykmeans==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(x[ykmeans==1,0], x[ykmeans==1,1], s=50, c='blue', label='Cluster 2')
plt.scatter(x[ykmeans==2,0], x[ykmeans==2,1], s=50, c='green', label='Cluster 3')
plt.scatter(x[ykmeans==3,0], x[ykmeans==3,1], s=50, c='cyan', label='Cluster 4')
plt.scatter(x[ykmeans==4,0], x[ykmeans==4,1], s=50, c='pink', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='yellow', label='Centroid')
plt.legend()
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()