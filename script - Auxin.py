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

import sklearn_relief as relief

#%%
def var_filter(df, n_fts=2000):
    sele_idx = df.var().nlargest(n_fts, keep='all').index
    df = df.loc[:, sele_idx]
    return df


def time_corr_filter(df, n_fts=400):
    y = np.array([168, 204, 216, 228, 240] * 5)
    corr = np.corrcoef(df.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-n_fts:]  # positively or negatively correlated.
    final = df.iloc[:, corr_idx]
    return final


def read_horm_data() -> pd.DataFrame:
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
            .str.replace("New_HDNTI", "HDNTII")
    return df_rt


def y_filter_supervised(df, y, n_fts=400):
    y = y.reindex(df.index)
    y = y.iloc[:, -1]
    corr = np.corrcoef(df.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-n_fts:]  # positively or negatively correlated.
    final = df.iloc[:, corr_idx]
    return final

def y_relieff_filter(df, y, n_fts=400):
    print("Start RefliefF...")
    y = y.reindex(df.index)
    y = y.iloc[:, -1]
    r = relief.RReliefF(n_features=n_fts, n_jobs=1)
    final = r.fit_transform(df, y)
    return final

#%%
def main():
#%% take log and select the first sample
    data = pd.read_excel("./for_proposal/Normalized_counts_Auxin RNASeq.xlsx", sheet_name=1, index_col=0)
    # data = pd.read_excel("./for_proposal/debug.xlsx", sheet_name=0, index_col=0)
    data = pd.DataFrame(np.clip(np.log2(data), 0, None), index=data.index, columns=data.columns)  # log transform the counts. There are many 0s, cliped to be 0.
    # sample = data.loc[data.index.str.startswith("LOC_Os06g")].T
    sample = data.T

    #%%  normalization 和 feature selection 整个dataset 一起做
    sample = var_filter(sample, 2000)
    ydf = read_horm_data()  # supervised. Followed by the unsupervised with generate the new vedios so not working (both 2000).
    sample = y_relieff_filter(sample, ydf, 400)  # Supervised
    sample.columns = sample.columns.str.replace("LOC_", "")
    #%%
    import pickle
    from refined.io import check_path_exists
    out_dir = 'for_proposal/relieff_y/'
    check_path_exists(out_dir)
    rfd = Refined(verbose=True, seed=2021)
    rfd.fit(sample, output_dir=out_dir)  # fit refined
    with open(out_dir + "RFD400.pickle", 'wb') as f:  # save object
        pickle.dump(rfd, f)
    rfd.plot_mapping()  # genes mapping
    sample = normalize_df(sample)  # normalize it overall, keep identical with the one before.
    rfd.generate_image(sample, output_folder=out_dir, normalize_feature=False)
    #%% Enclarge the npy imgs
    f_list = os.listdir(out_dir + "RFD_Images/")
    f_list = [os.path.join(out_dir + "RFD_Images/", x) for x in f_list]
    enlarge_images(f_list, text_groups=sample.index.tolist(), cm=cmapy.cmap('Blues_r'))  # 转换成png
    #%% 拼图
    img_list = []
    for root, dirs, files in os.walk(out_dir, topdown=False):
        for name in files:
            if name.endswith(".npy.png"):
                print(os.path.join(root, name))
                img_list.append(os.path.join(root, name))
    concat_images_2d(out_dir + "auto_concat.png", img_list)

#%%
if __name__ == "__main__":
    main()