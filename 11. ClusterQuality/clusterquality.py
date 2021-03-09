import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score

#%% Create data

data = datasets.make_blobs(2000, centers = 6)

X = data[0]
y = data[1]

#%% Plot data

fig, ax = plt.subplots(1, figsize = (15, 10))
ax.scatter(X[:, 0], X[:, 1], c = None, cmap = plt.cm.Set1, edgecolor = 'k')
plt.title('Generated dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('data.png')
plt.show()

#%% Execute

inerts = []
silhs = []
ys = []
clusternums = np.arange(3, 10, 1)

plt.figure(figsize = (20, 20))
ax = []

for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    inerts.append(clf.inertia_)
    silhs.append(silhouette_score(X, clf.predict(X)))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'Feature 1', ylabel = 'Feature 2')

plt.tight_layout()
plt.savefig('data_kmeans.png')
plt.show()

#%% Elbow method

plt.figure(figsize = (10, 6))
plt.plot(clusternums, inerts, 'rx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('inertia.png')
plt.show()

#%% Number of clusters

plt.figure(figsize = (10, 6))
plt.plot(clusternums, silhs, 'rx-')
plt.title('Silhouette score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.savefig('silh.png')
plt.show()