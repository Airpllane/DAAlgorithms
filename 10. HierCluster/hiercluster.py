import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

#%%

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#%% Load data 2

'''
mitbih_test.csv - ECG Heartbeat Categorization Dataset
'''
data2 = pd.read_csv('../8. SepCluster/mitbih_test.csv', header = None)
#data2 = data2.drop(np.random.choice(data2.index, 20000, replace = False))

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

#%% Execute

fig, ax = plt.subplots(4, 2, figsize = (20, 20))

linkages = np.array(['single', 'complete', 'average', 'ward'])
for linkage in linkages:
    i = np.where(linkages == linkage)[0][0]
    clf = AgglomerativeClustering(compute_distances = True, n_clusters = 5, linkage = linkage)
    y = clf.fit_predict(X)
    plot_dendrogram(clf, truncate_mode = 'level', p = 3, ax = ax[i, 0])
    ax[i, 1].scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
    ax[i, 0].set(xlabel = 'Number of points in node (or index of point if no parenthesis).')
    ax[i, 0].set_yticks([])
    ax[i, 1].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('linkages.png')
plt.show()