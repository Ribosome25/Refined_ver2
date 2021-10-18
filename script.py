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
def gene_filter_unsupervised(df, fold=0):
    # the first implementation: largest var filter and abs corr to temporal sequences
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


def read_auxin_data() -> pd.DataFrame:
    df = pd.read_excel("./for_proposal/Auxin_quantification_results_Jaspinder.xlsx")
    df.index = df['Treatment'] + df['Stage'].astype(str)
    result_list = []
    for each_condition in df.index.unique():
        group = df.loc[each_condition]
        avg = group.iloc[:, -1].mean()
        result_list.append([group.iloc[0, 0], group.iloc[0, 1], avg])
    df_rt = pd.DataFrame(result_list)
    df_rt.index = df_rt.iloc[:, 1].astype(str) + df_rt.iloc[:, 0]
    df_rt.index = df_rt.index.str.replace("Control", "C")\
        .str.replace("HDNT", "HDNTI")\
            .str.replace("NEW_HDNTI", "HDNTII")
    return df_rt


def gene_filter_supervised(df, y, fold=0):
    # Max var filter
    sele_idx = df.var().nlargest(2000, keep='all').index
    df = df.loc[:, sele_idx]
    # Max corr filter
    if fold is None:
        # None for selecting as a whole.
        control_group = df
        y = y.reindex(control_group.index)
    else:
        # if fold is int, normalize and select independently.
        control_group = df.iloc[5*fold: 5+5*fold]
        y = y.reindex(control_group.index)
    y = y.iloc[:, -1]
    corr = np.corrcoef(control_group.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-400:]  # positively or negatively correlated.

    final = control_group.iloc[:, corr_idx]
    return final

#%% take log and select the first sample
data = pd.read_excel("./for_proposal/Normalized_counts_Auxin RNASeq.xlsx", sheet_name=1, index_col=0)
data = pd.DataFrame(np.clip(np.log2(data), 0, None), index=data.index, columns=data.columns)  # log transform the counts. There are many 0s, cliped to be 0.
sample = data.loc[data.index.str.startswith("LOC_Os06g")].T

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
sample = gene_filter_unsupervised(sample, None)
ydf = read_auxin_data()  # supervised
sample = gene_filter_supervised(sample, ydf, None)  # Supervised
sample.columns = sample.columns.str.replace("LOC_Os01g", "")
#%%
import pickle
from refined.io import check_path_exists
check_path_exists('for_proposal/auxin_6/')
rfd = Refined(verbose=False)
rfd.fit(sample, output_dir='for_proposal/auxin_6/')  # fit refined
with open("for_proposal/auxin_6/RFD400.pickle", 'wb') as f:  # save object
    pickle.dump(rfd, f)
rfd.plot_mapping()  # genes mapping
sample = normalize_df(sample)  # normalize it overall, otw become outwashing bright.
#%%
for fold in range(0, 5):  # 保存图片。默认npy
    check_path_exists('for_proposal/auxin_6/{}/'.format(fold))
    final = sample.iloc[5*fold: 5+5*fold]
    rfd.generate_image(final, output_folder='for_proposal/auxin_6/{}/'.format(fold), normalize_feature=False)
    f_list = os.listdir('for_proposal/auxin_6/{}/RFD_Images'.format(fold))
    f_list = [os.path.join('for_proposal/auxin_6/{}/RFD_Images'.format(fold), x) for x in f_list]
    enlarge_images(f_list, text_groups=final.index.tolist(), cm=cmapy.cmap('Blues_r'))  # 转换成png

