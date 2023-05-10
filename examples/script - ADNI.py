# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:46:19 2021

@author: Ruibo

ADNI 

"""
import os
from matplotlib import pyplot as plt
from myToolbox.Stat import normalize_df
import numpy as np
import pandas as pd
import seaborn as sns
from myToolbox.Io import check_path_exists, read_df
from refined.refined import Refined
from refined.write_videos import enlarge_images, concat_images_2d
import cmapy
import pickle

from refined.feature_selection import var_filter

#%%

#%%
# def main():
if __name__ == "__main__":
#%% take log and select the first sample
    try:
        data = read_df("data/ADNI/ADNIClassData.parquet")
    except FileNotFoundError:
        data = pd.read_csv("data/ADNI/ADNIClassData.csv", index_col=0)
        data.to_parquet("data/ADNI/ADNIClassData.parquet", engine='fastparquet', compression='gzip')
    # sns.distplot(data.values)
    # plt.show()

    sample = data.T

    #%%  normalization 和 feature selection 整个dataset 一起做
    sample = var_filter(sample, 100)

    #%%
    out_dir = 'output/ADNI/'
    rfd = Refined(verbose=True, seed=22, working_dir=out_dir, assignment='lap')
    rfd.fit(sample)  # fit refined
    with open(out_dir + "RFD_100Genes_RS22.pickle", 'wb') as f:  # save object
        pickle.dump(rfd, f)
    rfd.plot_mapping()  # genes mapping
    # sample = normalize_df(sample)  # normalize it overall, keep identical with the one before.
    rfd.generate_image(sample, normalize_feature=False)
    rfd.save_mapping_to_csv()
    rfd.save_mapping_to_json()

    #%% Enclarge the npy imgs
    f_list = os.listdir(out_dir + "RFD_Images/")
    f_list = [os.path.join(out_dir + "RFD_Images/", x) for x in f_list]
    enlarge_images(f_list, text_groups=sample.index.tolist(), cm=cmapy.cmap('Blues_r'))  # 转换成png
    #%% 拼图
    # img_list = []
    # for root, dirs, files in os.walk(out_dir, topdown=False):
    #     for name in files:
    #         if name.endswith(".npy.png"):
    #             print(os.path.join(root, name))
    #             img_list.append(os.path.join(root, name))
    # concat_images_2d(out_dir + "auto_concat.png", img_list)

#%%
if __name__ == "__main__":
    main()