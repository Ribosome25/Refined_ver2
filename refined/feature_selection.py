"""
Select features based unsupervisly or supervisely. 

"""
import numpy as np

def var_filter(df, n_fts=2000):
    sele_idx = df.var().nlargest(n_fts, keep='all').index
    df = df.loc[:, sele_idx]
    return df


def pseudo_corr_filter(df, y=None, n_fts=400):
    if y is None:  # Pseudo time seqeunce
        y = np.arange(len(df))
    corr = np.corrcoef(df.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-n_fts:]  # positively or negatively correlated.
    final = df.iloc[:, corr_idx]
    return final

def y_filter_supervised(df, y, n_fts=400):
    y = y.reindex(df.index)
    y = y.iloc[:, -1]
    corr = np.corrcoef(df.T, y)[:-1, -1]
    corr_idx = np.argsort(abs(corr))[-n_fts:]  # positively or negatively correlated.
    final = df.iloc[:, corr_idx]
    return final

def y_relieff_filter(df, y, n_fts=400):
    import sklearn_relief as relief
    print("Start RefliefF...")
    y = y.reindex(df.index)
    y = y.iloc[:, -1]
    r = relief.RReliefF(n_features=n_fts, n_jobs=1)
    final = r.fit_transform(df, y)
    return final
