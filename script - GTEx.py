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


#%% take log and select the first sample
data = pd.read_table("G:/Datasets/GTEx/gene_tpm/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct", header=2, index_col=0).T
print("Data loaded")
# data = pd.read_csv("G:/Datasets/GTEx/gene_tpm/preview.csv", index_col=1, header=2).T
data = data.iloc[1:]
print(data.columns.duplicated().sum())
data.to_parquet("G:/Datasets/GTEx/gene_tpm/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.parquet")
sample = gene_filter_unsupervised(data, 2500)
del data
#%%
import pickle
from refined.io import check_path_exists
rfd_dir = "G:/Datasets/GTEx/RFD/"
check_path_exists(rfd_dir)
rfd = Refined(verbose=True)
rfd.fit(sample, output_dir=rfd_dir)  # fit refined
with open(rfd_dir + "RFD_MDS_2500.pickle", 'wb') as f:  # save object
    pickle.dump(rfd, f)
rfd.plot_mapping()  # genes mapping
sample = normalize_df(sample)  # normalize it overall, otw become outwashing bright.
# raise
#%%
rfd.generate_image(final, output_folder=rfd_dir, normalize_feature=False)
f_list = os.listdir(rfd_dir + 'RFD_Images'.format(fold))
f_list = [os.path.join(rfd_dir + 'RFD_Images', x) for x in f_list]
enlarge_images(f_list, text_groups=final.index.tolist(), cm=cmapy.cmap('Blues_r'))  # 转换成png
#%%
rfd.generate_image(final, output_folder=rfd_dir, normalize_feature=False, img_format='bmp')

