# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:40:02 2020

@author: Ruibo

Scoring
Count correct ordering (Quadruple) or correct adjacency


Usage:
        write_csv('{},{},{},{},{}'.format(distance,
                                      *CountSequence.count_cycles2(path, n=17),
                                      *CountSequence.count_cycles_adj(path, n=17)))
                                      
        write_csv('{},{},{},{},{}'.format(distance,
                                      *CountSequence.count_cycles2(path, n=12),
                                      *CountSequence.count_cycles_adj(path, n=12)))
"""
from itertools import combinations
import numpy as np
import warnings
import random
random.seed(2020)
#%%
def _sorted(x):
    return sorted(x) == list(x)

def iterSample(iterable, samplesize):
    results = []

    for i, v in enumerate(iterable):
        r = random.randint(0, i)
        if r < samplesize:
            if i < samplesize:
                results.insert(r, v) # add first samplesize items in random order
            else:
                results[r] = v # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")

    return results

def _sorted_one_side(each):
    doubled = list(each)
    doubled.extend(doubled)
    imin = doubled.index(min(each))
    selected = doubled[imin:imin+4]
    return sorted(selected)==selected

def count_one_side(t):
    if len(t) > 150:
        ids = list(range(len(t)))
        ids = sorted(random.sample(ids, 150))
        t = [t[ii] for ii in ids]
    cmbs = combinations(t, 4)
    cmbs = list(cmbs)
    if len(cmbs) > 1000000:
        ids = [x for x in range(len(cmbs))]
        random.shuffle(ids)
        ids = ids[:1000000]
        cmbs = [cmbs[ii] for ii in ids]
    ct = 0
    ln = 0
    for each in cmbs:
        ln += 1
        if _sorted_one_side(each):
            ct += 1
    return ct, ln

def count_two_sides(t):
    rt = [-x for x in t]
    ct, ln = count_one_side(t)
    rct, rln = count_one_side(rt)
    return ct+rct, ln

def count_cycles(t, n=7):
    t1 = [x for x in t if x < n]
    t2 = [x for x in t if x > n -1 ]
    c1, l1 = count_two_sides(t1)
    c2, l2 = count_two_sides(t2)
    return c1 + c2, l1+l2

def count_cycles2(t, n=7):
    if (len(t)%n != 0) or n>len(t):
        warnings.warn("n doesn't match with path list. Check n.")
    t2 = []
    for each in t:
        t2.append(each % n)
    c2, l2 = count_two_sides(t2)
    return c2, l2

def count_cycles_adj(t, n=7):
    if (len(t)%n != 0) or n>len(t):
        warnings.warn("n doesn't match with path list. Check n.")
    t2 = []
    for each in t:
        t2.append(each % n)
    ct = 0
    l2 = len(t2)
    for i in range(len(t2)):
        cmp = np.abs(t2[i-1] - t2[i])
        if cmp < 2 or cmp == n-1:
            ct += 1
    return ct, l2

if __name__ == '__main__':
    import random
    t1 = [x for x in range(0, 14)]
    t2 = [x for x in range(0, 14)]
    random.shuffle(t1)

    # a = count_cycles(t1)
    # b = count_cycles(t2)

    p = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 6, 7, 13]
    c, l  = count_cycles2(p*20)