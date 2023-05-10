# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:46:19 2021

@author: Ruibo

Do refined, gen video in one script.

Need to run from the root path. Move it out before running.

"""
import os
from myToolbox.Stat import normalize_df
import numpy as np
import pandas as pd
import seaborn as sns
from refined.io import check_path_exists
from refined.refined import Refined
from refined.write_videos import enlarge_images, concat_images_2d
import cmapy

import pickle
from refined.io import check_path_exists
#%%
def gene_filter_unsupervised(df, n=2500):
    # the first implementation: largest var filter and abs corr to temporal sequences
    # Max var filter
    sele_idx = df.var().nlargest(n, keep='all').index
    final = df.loc[:, sele_idx]
    # Max corr filter
    # if fold is None:
        ## None for selecting as a whole.
        # control_group = df
        # y = np.array([168, 204, 216, 228, 240] * 5)
    # else:
        ## if fold is int, normalize and select independently.
        # control_group = df.iloc[5*fold: 5+5*fold]
        # y = np.array([168, 204, 216, 228, 240])
    # corr = np.corrcoef(control_group.T, y)[:-1, -1]
    # corr_idx = np.argsort(abs(corr))[-400:]  # positively or negatively correlated.

    # final = control_group.iloc[:, corr_idx]
    return final

# lookup_table = pd.read_csv("G:/Datasets/CCRCC/CPTAC_CCRCC_Transcriptome_rpkm/RNA_clinical.csv", index_col=2, header=None)
# lookup_table[1] = lookup_table[1].replace("Primary Tumor", 'T').replace("Solid Tissue Normal", "N")
# lookup_table["c"] = lookup_table[0] + "-" + lookup_table[1]

#%% 
data = pd.read_table("G:/Datasets/CCRCC/Supplementary_Data_Proteome_DIA.csv", index_col=0, header=0, sep=',')
data = data.iloc[:, 3:].T
data = data.loc[:, ~data.columns.duplicated()]
assert(data.columns.duplicated().sum() == 0)
data.fillna(data.mean(), inplace=True)
data.dropna(axis=1, how='all', inplace=True)
data.isna().sum().sum()
# new_idx = lookup_table.reindex(data.index)['c']
# data.index = new_idx
sample = gene_filter_unsupervised(data, 2500)

#%%
REP = 10
for ii in range(REP):
    rfd_dir = "G:/Datasets/CCRCC/prot_dia/{}/".format(ii)
    check_path_exists(rfd_dir)
    rfd = Refined(dim_reduction='MDS', distance_metric='correlation',
                 assignment='lap', verbose=True, seed=ii)
    rfd.fit(sample, output_dir=rfd_dir)  # fit refined
    with open(rfd_dir + "RFD_MDS_2500.pickle", 'wb') as f:  # save object
        pickle.dump(rfd, f)
    rfd.plot_mapping()  # genes mapping
    # sample = normalize_df(sample)  # normalize it overall, otw become outwashing bright.
    rfd.generate_image(sample, output_folder=rfd_dir)
#%%

