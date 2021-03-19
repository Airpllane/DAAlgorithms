import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#%% Load data

'''
banknotes.csv - Swiss banknote counterfeit detection
'''
data1 = pd.read_csv('../12. Outliers/banknotes.csv').iloc[:110]
del data1['conterfeit']

'''
winequality-red.csv - Red Wine Quality
'''
data2 = pd.read_csv('../3. Bayes/winequality-red.csv')
del data2['quality']

#%% Process data 1

X = data1
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data 1

plt.figure(figsize = (8, 8))
plt.scatter(X[:, 0], X[:, 1], marker = 'x')
plt.title('Banknotes')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
#plt.savefig('Original1.png')
plt.show()