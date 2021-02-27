import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter

from fim import apriori, fpgrowth, eclat

#%%

def avg_len(lst):
    '''
    Average list length

    Parameters
    ----------
    lst : list
        List of elements with measurable length
    
    Returns
    -------
    float 
        Average length of a list element
        
    '''
    lens = [len(i) for i in lst]
    return 0 if len(lens) == 0 else (float(sum(lens)) / len(lens))

def wrap_apriori(data_list, min_sup, min_len):
    '''
    Executes Apriori algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.

    Returns
    -------
    results : list
        Results returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(apriori(data_list, supp = min_sup, zmin = min_len, report = 's'))
    ex_time = time.time() - start_time
    return results, ex_time

def wrap_fp(data_list, min_sup, min_len):
    '''
    Executes FP Growth algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.

    Returns
    -------
    results : list
        Records returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(fpgrowth(data_list, supp = min_sup, zmin = min_len, report = 's'))
    ex_time = time.time() - start_time
    return results, ex_time

def wrap_eclat(data_list, min_sup, min_len):
    '''
    Executes ECLAT algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.

    Returns
    -------
    results : list
        Records returned by the algorithm.
    ex_time : float
        Execution time.
        
    '''
    start_time = time.time()
    results = list(eclat(data_list, supp = min_sup, zmin = min_len, report = 's'))
    ex_time = time.time() - start_time
    return results, ex_time

def execute_algs(data_list, min_sup, min_len = 1):
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
    apriori_results, apriori_time = wrap_apriori(data_list, min_sup, min_len)
    fp_results, fp_time = wrap_fp(data_list, min_sup, min_len)
    eclat_results, eclat_time = wrap_eclat(data_list, min_sup, min_len)
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
        print('Set: ' + str(record[0]) + '\nSupport: ' + str(record[1]))
        
#%% Datasets

'''
groceries.csv - Groceries dataset
'''
data1 = pd.read_csv('groceries.csv')

'''
store_data.csv - Dataset for Apriori Algorithm - Frequent Itemsets
'''
data2 = pd.read_csv('store_data.csv', header = None)

'''
AnimeList.csv - MyAnimeList Dataset
'''
data3 = pd.read_csv('AnimeList.csv')

#%% Data to lists

data1_list = list(data1.groupby(['Member_number', 'Date'])['itemDescription'].apply(set).apply(list))

data2_list = [[i for i in j if i == i] for j in data2.values.tolist()]

data3_list = list(data3['genre'].str.split(', ').dropna())

#%% Select

data_list = data3_list
min_len = 1

#%%
'''
result, _ = wrap_apriori(data_list, 0.39, 5)
print_results(result)
'''
#%% Execute

supports = np.arange(0.5, 10, 0.1)
atimes = []
ftimes = []
etimes = []
num_sets = []
longest_set = []
lens_sets = []

for min_sup in supports:
    results, times = execute_algs(data_list, min_sup)
    atimes.append(times['Apriori'])
    ftimes.append(times['FP'])
    etimes.append(times['ECLAT'])
    if len(results['Apriori']) == 0:
        num_sets.append(0)
        longest_set.append(0)
        lens_sets.append({})
        continue
    num_sets.append(len(results['Apriori']))
    longest_set.append(max([len(i[0]) for i in results['Apriori']]))
    lens_sets.append(dict(Counter([len(i[0]) for i in results['Apriori']])))

#%% Plot

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (15, 10))

ax[0, 0].plot(supports, atimes)
ax[0, 0].plot(supports, ftimes)
ax[0, 0].plot(supports, etimes)
ax[0, 0].legend(['Apriori', 'FP growth', 'ECLAT'])
ax[0, 0].set_title('Time taken')
ax[0, 0].set(xlabel = 'Support, %', ylabel = 'Time, s')

ax[0, 1].plot(supports, num_sets)
ax[0, 1].set_title('Total number of sets')
ax[0, 1].set(xlabel = 'Support, %', ylabel = 'Sets')

ax[1, 0].plot(supports, longest_set)
ax[1, 0].set_title('Longest set')
ax[1, 0].set(xlabel = 'Support, %', ylabel = 'Items')

sets_of_len = {}
for i in range(1, max(longest_set) + 1):
        sets_of_len[i] = []
for i in lens_sets:
    for j in range(1, max(longest_set) + 1):
        sets_of_len[j].append(i.get(j, 0))

sol_legend = []
for i in range(1, max(longest_set) + 1):
    ax[1, 1].plot(supports, sets_of_len.get(i, []))
    sol_legend.append(str(i))
ax[1, 1].legend(sol_legend)
ax[1, 1].set_title('Sets of specific length')
ax[1, 1].set(xlabel = 'Support, %', ylabel = 'Sets')
plt.savefig('datat.png')
plt.show()

#%% Plot times

'''
plt.plot(supports, atimes)
plt.plot(supports, ftimes)
plt.plot(supports, etimes)
plt.legend(['Apriori', 'FP growth', 'ECLAT'])
plt.title('Time taken')
plt.xlabel('Support, %')
plt.ylabel('Time, s')
plt.savefig('times.png')
plt.show()
'''