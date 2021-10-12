"""
Find some highly correlated ones, and some negatively correlated ones.

Do the refined.

"""
import numpy as np
import pandas as pd
import seaborn as sns

#%%
data = pd.read_excel("Normalized_counts_Auxin RNASeq.xlsx", sheet_name=1, index_col=0)

sample = data.loc[data.index.str.startswith("LOC_Os01g")].T

sample.to_parquet("splited_data/s1_all.parquet", engine='fastparquet', compression='gzip')
sample.iloc[:5].to_parquet("splited_data/s1_C.parquet", engine='fastparquet', compression='gzip')
sample.iloc[:5, :500].to_parquet("splited_data/s1_C_test.parquet", engine='fastparquet', compression='gzip')

