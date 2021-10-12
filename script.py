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
from refined.Write_videos import enlarge_images
import cmapy
#%%
def gene_filter(df, fold=0):
    # Max var filter
    sele_idx = df.var().nlargest(2000, keep='all').index
    df = df.loc[:, sele_idx]
    # Max corr filter
    if fold is None: 
        # None for selecting as a whole.
        control_group = df
        y = np.array([168, 204, 216, 228, 240] * 5)
    else:
        # if fold is int, normalize and select independently. 
        control_group = df.iloc[5*fold: 5+5*fold]
        y = np.array([168, 204, 216, 228, 240])
    corr = np.corrcoef(control_group.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-400:]  # positively or negatively correlated.

    final = control_group.iloc[:, corr_idx]
    return final

#%% take log and select the first sample
data = pd.read_excel("./for_proposal/Normalized_counts_Auxin RNASeq.xlsx", sheet_name=1, index_col=0)
data = pd.DataFrame(np.clip(np.log2(data), 0, None), index=data.index, columns=data.columns)  # log transform the counts. There are many 0s, cliped to be 0. 
sample = data.loc[data.index.str.startswith("LOC_Os01g")].T

#%%  Individual 每个condition 分别做 normalization 和 feature selection
# for fold in range(0, 5):
#     check_path_exists('for_proposal/{}/'.format(fold))
#     final = gene_filter(sample, fold)
#     rfd = Refined(verbose=False)
#     rfd.fit(final, output_dir='for_proposal/{}/'.format(fold))
#     print("Fitted")
#     rfd.generate_image(final, output_folder='for_proposal/{}/'.format(fold))
#     f_list = os.listdir('for_proposal/{}/RFD_Images'.format(fold))
#     f_list = [os.path.join('for_proposal/{}/RFD_Images'.format(fold), x) for x in f_list]
#     enlarge_images(f_list, text_groups=final.index.tolist(), cm=cmapy.cmap('Blues_r'))
#%%  normalization 和 feature selection 整个dataset 一起做
sample = gene_filter(sample, None)
rfd = Refined(verbose=False)
rfd.fit(sample, output_dir='for_proposal/')
sample = normalize_df(sample)  # normalize it overall, otw become outwashing bright. 

for fold in range(0, 5):
    check_path_exists('for_proposal/{}/'.format(fold))
    final = sample.iloc[5*fold: 5+5*fold]
    rfd.generate_image(final, output_folder='for_proposal/{}/'.format(fold), normalize_feature=False)
    f_list = os.listdir('for_proposal/{}/RFD_Images'.format(fold))
    f_list = [os.path.join('for_proposal/{}/RFD_Images'.format(fold), x) for x in f_list]
    enlarge_images(f_list, text_groups=final.index.tolist(), cm=cmapy.cmap('Blues_r'))