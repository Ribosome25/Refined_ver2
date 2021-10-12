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
#%%
# Max var filter
sele_idx = sample.var().nlargest(2000, keep='all').index
sample = sample.loc[:, sele_idx]
# Max corr filter
control_group = sample.iloc[:5]
y = np.array([1, 2, 3, 4, 5])
corr = np.corrcoef(control_group.T, y)[:-1, -1]
corr_idx = np.argsort(corr)[-400:]

final = control_group.iloc[:, corr_idx]
#%%
final.to_parquet("splited_data/s1_C_slec.parquet", engine='fastparquet', compression='gzip')
raise
#%%
sample.to_parquet("splited_data/s1_all.parquet", engine='fastparquet', compression='gzip')
sample.iloc[:5].to_parquet("splited_data/s1_C.parquet", engine='fastparquet', compression='gzip')
sample.iloc[:5, :500].to_parquet("splited_data/s1_C_test.parquet", engine='fastparquet', compression='gzip')

