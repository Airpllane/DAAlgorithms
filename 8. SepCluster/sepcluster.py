import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

#%% Load data
'''
mobiletest.csv - Mobile Price Classification
'''
data1 = pd.read_csv('mobiletest.csv')
target1 = data1['price_range']
del data1['price_range']

#%% Process data

X = data1
y = target1
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data

fig, ax = plt.subplots(1, figsize = (10, 10))
ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.title('Original dataset')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
plt.savefig('Data1.png')
plt.show()

#%% Execute

ys = []
clusternums = np.arange(3, 10, 1)
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()

plt.savefig('Data1_KMeans.png')
plt.show()

#%% Add noise

for row in X:
    if(np.random.randint(11) < 3):
        col = np.random.randint(2)
        row[col] += (np.random.randint(-1, 2) * 0.1 * row[col]) 

#%% Execute (KMeans)

ys = []
clusternums = np.arange(3, 10, 1)
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('Noise_KMeans.png')
plt.show()

#%% Execute (KMedoids)


ys = []
clusternums = np.arange(3, 10, 1)
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMedoids(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('Noise_KMedoids.png')
plt.show()


#%% Load data 2

'''
mitbih_test.csv - ECG Heartbeat Categorization Dataset
'''
data2 = pd.read_csv('mitbih_test.csv', header = None)

#%% Process data 2

X = data2
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data

fig, ax = plt.subplots(1, figsize = (10, 10))
ax.scatter(X[:, 0], X[:, 1], c = None, cmap = plt.cm.Set1, edgecolor = 'k')
plt.title('Original dataset')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
plt.savefig('Data2.png')
plt.show()

#%% Execute (KMeans)

ys = []
clusternums = np.arange(3, 10, 1)
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('Data2_KMeans.png')
plt.show()

#%% Execute (KMedoids)

ys = []
clusternums = np.arange(3, 10, 1)
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMedoids(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i].set_title('For ' + str(clusternum) + ' clusters')
    ax[i].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('Data2_KMedoids.png')
plt.show()
