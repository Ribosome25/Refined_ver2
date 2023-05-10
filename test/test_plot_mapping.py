# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Run from root. 
Move it out before running. 

lagacy. Not working on 7/2/22


"""
import pandas as pd
import pickle

from refined.io import check_path_exists
from refined.refined import Refined

if __name__ == "__main__":
    sample = pd.read_csv("./test/test.csv", index_col=0).T
    rfd_dir = "./test/test_plot_mapping/"
    check_path_exists(rfd_dir)
    rfd = Refined(dim_reduction='MDS', distance_metric='correlation',
                 assignment='lap', verbose=True, seed=0)
    rfd.fit(sample, output_dir=rfd_dir)  # fit refined
    with open(rfd_dir + "RFD_MDS.pickle", 'wb') as f:  # save object
        pickle.dump(rfd, f)
    rfd.plot_mapping(output_dir=rfd_dir)  # genes mapping
    # sample = normalize_df(sample)  # normalize it overall, otw become outwashing bright.
    rfd.generate_image(sample, img_format='png', output_folder=rfd_dir)
    rfd.save_mapping_to_csv()
