import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter

from apyori import apriori
import pyfpgrowth
from pyECLAT import ECLAT



#%% Functions

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
    apriori_records : list
        Records returned by the algorithm.
    apriori_time : float
        Execution time.
        
    '''
    start_time = time.time()
    apriori_records = list(apriori(data_list, min_support = min_sup))
    apriori_records = list(filter(lambda x: len(x.items) >= min_len, apriori_records))
    apriori_time = time.time() - start_time
    return apriori_records, apriori_time

def wrap_fp(data_list, min_sup):
    '''
    Executes FP Growth algorithm

    Parameters
    ----------
    data_list : list 
        List of transactions
    min_sup : float
        Min support value.

    Returns
    -------
    fp_patterns : dict
        Records returned by the algorithm.
    fp_time : float
        Execution time.

    '''
    min_occ = int(len(data_list) * min_sup)
    start_time = time.time()
    fp_patterns = pyfpgrowth.find_frequent_patterns(data_list, min_occ)
    #fp_rules = pyfpgrowth.generate_association_rules(fp_patterns, min_conf)
    fp_time = time.time() - start_time
    return fp_patterns, fp_time

def wrap_eclat(data_frame, min_sup, min_len):
    '''
    Executes ECLAT algorithm

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame of transactions.
    min_sup : float
        Min support value.
    min_len : int
        Min length of a set.

    Returns
    -------
    eclat_patterns : dict
        Records returned by the algorithm.
    eclat_time : float
        Execution time.

    '''
    start_time = time.time()
    eclat_instance = ECLAT(data = data_frame, verbose = False)
    _, eclat_patterns = eclat_instance.fit(min_support = min_sup, 
                                                   min_combination = min_len, 
                                                   max_combination = 2,
                                                   separator = ', ',
                                                   verbose = False)
    eclat_time = time.time() - start_time
    return eclat_patterns, eclat_time

def print_apriori(records):
    '''
    Prints the results of Apriori algorithm to console

    Parameters
    ----------
    records : list
        Records returned by Apriori algorithm.

    '''
    print("--- Apriori ---")
    for record in records:
        print("Set: " + ', '.join([x for x in record[0]]))
        print("Support: " + str(record[1]))
        '''
        print("*****************")
        for rule in record[2]:
            print("Rule: " + str(list(rule[0])) + " -> " + str(list(rule[1])))
            print("Confidence: " + str(rule[2]))
            print("Lift: " + str(rule[3]))
            print("-----------------")
        '''

def print_fp(patterns):
    '''
    Prints the results of FP Growth algorithm to console

    Parameters
    ----------
    patterns : dict
        Records returned by FP Growth algorithm.

    '''
    print('--- FP growth ---')
    for pattern in patterns:
        if len(pattern) < min_len:
            continue
        print('Set: ' + str(', '.join(pattern)))
        print('Support: ' + str(patterns[pattern] / len(data_list)))
        
    '''
    for rule in fp_rules:
        print('Rule: ' + str(rule[0]) + ' -> ' + str(fp_rules[rule][0][0]))
        print('Confidence: ' + str(fp_rules[rule][1]))
    '''

def print_eclat(patterns):
    '''
    Prints the results of ECLAT algorithm to console

    Parameters
    ----------
    patterns : dict
        Records returned by ECLAT algorithm.

    '''
    print('--- ECLAT ---')
    for pattern in patterns:
        print('Set: ' + str(pattern))
        print('Support: ' + str(patterns[pattern]))

def execute_algs(data_list, min_sup, min_len):
    '''
    Executes Apriori, FP Growth and Eclat algorithms on a given list of transactions

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
    results : dict
        Contains results of all three algorithms.
    times : dict
        Contains execution times of all three algorithms.

    '''
    apriori_records, apriori_time = wrap_apriori(data_list, min_sup, min_len)
    fp_patterns, fp_time = wrap_fp(data_list, min_sup)
    eclat_patterns, eclat_time = wrap_eclat(pd.DataFrame(data_list), min_sup, min_len)
    results = {'Apriori': apriori_records, 'FP': fp_patterns, 'ECLAT': eclat_patterns}
    times = {'Apriori': apriori_time, 'FP': fp_time, 'ECLAT': eclat_time}
    return results, times

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

#%% Select list

data_list = data2_list

#%% Min length of a set

min_len = 1

#%% Algs

supports = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
atimes = []
ftimes = []
etimes = []
num_sets = []
longest_set = []
lens_sets = []

for min_sup in supports:
    results, times = execute_algs(data_list, min_sup, min_len)
    atimes.append(times['Apriori'])
    ftimes.append(times['FP'])
    etimes.append(times['ECLAT'])
    if len(results['Apriori']) == 0:
        num_sets.append(0)
        longest_set.append(0)
        lens_sets.append({})
        continue
    num_sets.append(len([list(j[0]) for j in results['Apriori']]))
    longest_set.append(max([len(list(j[0])) for j in results['Apriori']]))
    lens_sets.append(dict(Counter([len(list(j[0])) for j in results['Apriori']])))

#%% Plots

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (15, 10))

ax[0, 0].plot(supports, atimes)
ax[0, 0].plot(supports, ftimes)
ax[0, 0].plot(supports, etimes)
ax[0, 0].legend(['Apriori', 'FP growth', 'ECLAT'])
ax[0, 0].set_title('Time taken')
ax[0, 0].set(xlabel = 'Support', ylabel = 'Time, s')

ax[0, 1].plot(supports, num_sets)
ax[0, 1].set_title('Total number of sets')
ax[0, 1].set(xlabel = 'Support', ylabel = 'Sets')

ax[1, 0].plot(supports, longest_set)
ax[1, 0].set_title('Longest set')
ax[1, 0].set(xlabel = 'Support', ylabel = 'Items')

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
ax[1, 1].set(xlabel = 'Support', ylabel = 'Sets')
plt.savefig('data2.png')
plt.show()
