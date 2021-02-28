import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter

from fim import apriori, fpgrowth, eclat

#%% Functions

def wrap_apriori(data_list, min_sup, min_len, min_conf):
    '''
    Executes Apriori algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions.
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.
    min_conf : float
        Min confidence value.

    Returns
    -------
    results : list
        Results returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(apriori(data_list, target = 'r', supp = min_sup, zmin = min_len, conf = min_conf, report = 'c'))
    ex_time = time.time() - start_time
    return results, ex_time

def wrap_fp(data_list, min_sup, min_len, min_conf):
    '''
    Executes FP Growth algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions.
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.
    min_conf : float
        Min confidence value.
        
    Returns
    -------
    results : list
        Records returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(fpgrowth(data_list, target = 'r', supp = min_sup, zmin = min_len, conf = min_conf, report = 'c'))
    ex_time = time.time() - start_time
    return results, ex_time

def wrap_eclat(data_list, min_sup, min_len, min_conf):
    '''
    Executes ECLAT algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions.
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.
    min_conf : float
        Min confidence value.
        
    Returns
    -------
    results : list
        Records returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(eclat(data_list, target = 'r', supp = min_sup, zmin = min_len, conf = min_conf, report = 'c'))
    ex_time = time.time() - start_time
    return results, ex_time

def execute_algs(data_list, min_sup, min_conf, min_len = 1):
    '''
    Executes Apriori, FP Growth and Eclat algorithms on a given list of transactions

    Parameters
    ----------
    data_list : list 
        List of transactions.
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.

    Returns
    -------
    results : dict
        Contains results of all three algorithms.
    times : dict
        Contains execution times of all three algorithms.

    '''
    apriori_results, apriori_time = wrap_apriori(data_list, min_sup, min_len, min_conf)
    fp_results, fp_time = wrap_fp(data_list, min_sup, min_len, min_conf)
    eclat_results, eclat_time = wrap_eclat(data_list, min_sup, min_len, min_conf)
    results = {'Apriori': apriori_results, 'FP': fp_results, 'ECLAT': eclat_results}
    times = {'Apriori': apriori_time, 'FP': fp_time, 'ECLAT': eclat_time}
    return results, times

def print_results(results):
    '''
    Print algorithm results to console

    Parameters
    ----------
    results : list
        Results returned by an algorithm.

    '''
    for record in results:
        print('Rule: (' + str(' + '.join(record[1])) + ') -> ' + str(record[0]) + 
              '\nConfidence: ' + str(record[2]))

#%% Datasets

'''
groceries.csv - Groceries dataset
'''
data1 = pd.read_csv('../1.Itemset/groceries.csv')

'''
store_data.csv - Dataset for Apriori Algorithm - Frequent Itemsets
'''
data2 = pd.read_csv('../1.Itemset/store_data.csv', header = None)

'''
AnimeList.csv - MyAnimeList Dataset
'''
data3 = pd.read_csv('../1.Itemset/AnimeList.csv')

#%% Data to lists

data1_list = list(data1.groupby(['Member_number', 'Date'])['itemDescription'].apply(set).apply(list))

data2_list = [[i for i in j if i == i] for j in data2.values.tolist()]

data3_list = list(data3['genre'].str.split(', ').dropna())

#%% Select

data_list = data1_list
min_sup = 0.2
min_len = 2

#%%

'''
results = fpgrowth(data_list, target = 'r', supp = min_sup, zmin = min_len, report = 'c', conf = 80)
print_results(results)
'''

#%% Execute

confs = np.arange(1, 100, 5)
atimes = []
ftimes = []
etimes = []
num_rules = []
max_objs = []
num_rules_under7 = []

for min_conf in confs:
    results, times = execute_algs(data_list, min_sup, min_conf, min_len)
    atimes.append(times['Apriori'])
    ftimes.append(times['FP'])
    etimes.append(times['ECLAT'])
    if len(results['Apriori']) == 0:
        num_rules.append(0)
        max_objs.append(0)
        num_rules_under7.append(0)
        continue
    num_rules.append(len(results['Apriori']))
    max_objs.append(max([(1 if type(i[0]) == str else len(i[0])) + len(i[1]) for i in results['Apriori']]))
    num_rules_under7.append(sum(x <= 7 for x in [(1 if type(i[0]) == str else len(i[0])) + len(i[1]) for i in results['Apriori']]))

#%% Plot

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (15, 10))

ax[0, 0].plot(confs, atimes)
ax[0, 0].plot(confs, ftimes)
ax[0, 0].plot(confs, etimes)
ax[0, 0].legend(['Apriori', 'FP growth', 'ECLAT'])
ax[0, 0].set_title('Time taken')
ax[0, 0].set(xlabel = 'Confidence, %', ylabel = 'Time, s')

ax[0, 1].plot(confs, num_rules)
ax[0, 1].set_title('Total number of rules')
ax[0, 1].set(xlabel = 'Confidence, %', ylabel = 'Rules')

ax[1, 0].plot(confs, max_objs)
ax[1, 0].set_title('Max objects in rule')
ax[1, 0].set(xlabel = 'Confidence, %', ylabel = 'Items')

ax[1, 1].plot(confs, num_rules_under7)
ax[1, 1].set_title('Number of rules containing <= 7 objects')
ax[1, 1].set(xlabel = 'Confidence, %', ylabel = 'Items')

plt.savefig('data1.png')
plt.show()

#%%

'''
plt.plot(confs, num_rules_under7)
plt.title('Number of rules containing > 7 objects')
plt.xlabel('Confidence, %')
plt.ylabel('Items')
plt.savefig('7.png')
plt.show()
'''