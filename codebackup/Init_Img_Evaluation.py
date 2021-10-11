# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:32:42 2019

@author: Ruibzhan

init feature mappings,
using different dim-reduction methods,
only sqr mappings are considered..
saved file is in [feature_names_list,dist_mat,Img]
Img is a np array with object datatype.

"""
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.spatial import distance
import pickle

def InitCorr(Dist_mat, parent_pop, NN, BB=0, KK=0):
    # parent_pop 47x47, elements str F***
    # Dist_mat: original distance between feats, np.array?

    FFN = Dist_mat.shape[0] # number of features
    NN2 = NN*NN
    FF = [f'F{i}' for i in range(NN2)]
    FF_FFN = [f'F{i}' for i in range(FFN)]
    PP_MM = np.reshape([[0.0 for y in range(NN2)] for x in range(NN2)],(NN2,NN2))
    PP_mm = []
    for i in range(NN):
        for j in range(NN):
            PP_mm.append([i,j])
    pix_dist = distance.pdist(PP_mm)
    PP_MM = distance.squareform(pix_dist, checks=False)
    Dist_mat_Sq = distance.squareform(Dist_mat)
    parent_pop2 = parent_pop          ## Added by Omid
    parent_pop2_vec = np.reshape(parent_pop2,(1,NN2)).tolist()[0]
    Indd2_vec = [parent_pop2_vec.index(i) for i in FF_FFN]
    PP_MM2_vec = PP_MM[np.ix_(Indd2_vec,Indd2_vec)]
    CORRr1, P_VAL = pearsonr(Dist_mat_Sq,
                             distance.squareform(PP_MM2_vec, checks=False))
    #CORRr1 = Cor2_arr[1,0]
    print("Initial: >>>> ",CORRr1)
    return CORRr1

#%% Load original input
with open('./data/GDSC_10fold.pickle', 'rb') as file:
    data = pickle.load(file)
    rna = data['rna_seq']

gene = []
# gene.append(pd.read_csv('./data/HH.csv', index_col=0, header=None))
# gene.append(pd.read_csv('./data/NOTCH.csv', index_col=0, header=None))
# gene.append(pd.read_csv('./data/TGFB.csv', index_col=0, header=None))
gene.append(pd.read_csv('./data/Melanoma.csv', index_col=0, header=None))
kegg = pd.concat(gene, axis=0).iloc[:, 0].unique()

# kegg = pd.read_csv('./data/KEGG.csv', index_col=0).iloc[:50, 0]
# Filter for the KEGG pathway
original_input = rna.reindex(columns=kegg).dropna(axis=1)
#%% Load for Gaussian clustering
original_input = pd.read_csv('./Gauss_clusters/original_image/data.csv', index_col=0)
#%%
print(">>>> Data Loaded")
import math
feature_names_list = original_input.columns.tolist()
nn = math.ceil(np.sqrt(len(feature_names_list))) # image dimension
Nn = original_input.shape[1] # Feature amount total

dist_mat = 1 - (original_input.corr()) # dist = 1-correlation

from myToolbox.Stat import normalize_df
original_input = normalize_df(original_input)
transposed_input = original_input.T # dim-reduc, the fts as instances, data T as DF

#%% to 2-d representation (xy coordinates)
def two_d_norm(xy):
    # xy is N x 2 xy cordinates, returns normed-xy on [0,1]
    norm_xy = (xy - xy.min(axis = 0)) / (xy - xy.min(axis = 0)).max(axis = 0)
    return norm_xy

def two_d_eq(xy):
    # xy is N x 2 xy cordinates, returns eq-xy on [0,1]
    xx_rank = np.argsort(xy[:,0])
    yy_rank = np.argsort(xy[:,1])
    eq_xy = np.full(xy.shape,np.nan)
    for ii in range(xy.shape[0]):
        xx_idx = xx_rank[ii]
        yy_idx = yy_rank[ii]
        eq_xy[xx_idx,0] = ii * 1/Nn
        eq_xy[yy_idx,1] = ii * 1/Nn
    return eq_xy
#%% to pixels
def Assign_features_to_pixels(xy, nn, verbose = False):
    # For each unassigned feature, find its nearest pixel, repeat until every ft is assigned
    # xy is the 2-d coordinates (normalized to [0,1]); nn is the image width. Img size = n x n
    # generate the result summary table, xy pixels; 3rd is nan for filling the idx
    Nn = xy.shape[0]

    from itertools import product
    pixel_iter = product([x for x in range(nn)],repeat = 2)
    result_table = np.full((nn*nn,3),np.nan)
    ii = 0
    for each_pixel in pixel_iter:
        result_table[ii,:2] = np.array(each_pixel)
        ii+=1
    # Use numpy array for speed

    from sklearn.metrics import pairwise_distances

#    xy = eq_xy
    centroids = result_table[:,:2] / nn + 0.5/nn
    pixel_avail = np.ones(nn*nn).astype(bool)
    feature_assigned = np.zeros(Nn).astype(bool)

    dist_xy_centroids = pairwise_distances(centroids,xy,metric='euclidean')

    while feature_assigned.sum()<Nn:
        # Init the pick-relationship table
        pick_xy_centroids = np.zeros(dist_xy_centroids.shape).astype(bool)

        for each_feature in range(Nn):
            # for each feature, find the nearest available pixel
            if feature_assigned[each_feature] == True:
                # if this feature is already assigned, skip to the next ft
                continue
            else:
                # if not assigned:
                for ii in range(nn*nn):
                    # find the nearest avail pixel
                    nearest_pixel_idx = np.argsort(dist_xy_centroids[:,each_feature])[ii]
                    if pixel_avail[nearest_pixel_idx] == True:
                        break
                    else:
                        continue
                pick_xy_centroids[nearest_pixel_idx,each_feature] = True

        for each_pixel in range(nn*nn):
            # Assign the feature No to pixels
            if pixel_avail[each_pixel] == False:
                continue
            else:
                # find all the "True" features. np.where returns a tuple size 1
                related_features = np.where(pick_xy_centroids[each_pixel,:] == 1)[0]
                if len(related_features) == 1:
                    # Assign it
                    result_table[each_pixel,2] = related_features[0]
                    pixel_avail[each_pixel] = False
                    feature_assigned[related_features[0]] = True
                elif len(related_features) > 1:
                    related_dists = dist_xy_centroids[each_pixel,related_features]
                    best_feature = related_features[np.argsort(related_dists)[0]] # Sort, and pick the nearest one among them
                    result_table[each_pixel,2] = best_feature
                    pixel_avail[each_pixel] = False
                    feature_assigned[best_feature] = True
        if verbose:
            print(">> Assign features to pixels:", feature_assigned.sum(),"/",Nn)
    result_table = result_table.astype(int)

    img = np.full((nn,nn),'NaN').astype(object)
    for each_pixel in range(nn*nn):
        xx = result_table[each_pixel,0]
        yy = result_table[each_pixel,1]
        ft = 'F' + str(result_table[each_pixel,2])
        img[xx,yy] = ft
    return img.astype(object)
#%% Dim-reduction
import sklearn.manifold as mnf

mds = mnf.MDS(dissimilarity='precomputed')
mds_xy = mds.fit_transform(dist_mat)
#
# isomap = mnf.Isomap(n_neighbors=10, n_components=2, eigen_solver='dense', path_method= 'D', n_jobs=3)
# ism_xy = isomap.fit_transform(transposed_input)
#
# lle = mnf.LocallyLinearEmbedding(n_neighbors=15)
# lle_xy = lle.fit_transform(transposed_input)
#
# tsne = mnf.TSNE()
# tsne_xy = tsne.fit_transform(transposed_input)
#
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca_xy = pca.fit_transform(transposed_input)

# import pydiffmap as dm
# dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
#                                                   alpha = 0.5, epsilon = 1, k = 32)
# dm_xy = dfmap.fit_transform(transposed_input)

print(">>>> Dim-reduction done")

#%%
print(">>>> MDS")
eq_xy = two_d_eq(mds_xy)
Img = Assign_features_to_pixels(eq_xy,nn,verbose=1)
InitCorr(dist_mat,Img,nn)

# print(">>>> IsoMap")
# eq_xy = two_d_eq(ism_xy)
# Img = Assign_features_to_pixels(eq_xy,nn)
# InitCorr(dist_mat,Img,nn)
#
# print(">>>> LLE")
# eq_xy = two_d_eq(lle_xy)
# Img = Assign_features_to_pixels(eq_xy,nn)
# InitCorr(dist_mat,Img,nn)

# print(">>>> PCA")
# eq_xy = two_d_eq(pca_xy)
# Img = Assign_features_to_pixels(eq_xy,nn)
# InitCorr(dist_mat,Img,nn)
#
# print(">>>> tSNE")
# eq_xy = two_d_eq(tsne_xy)
# Img = Assign_features_to_pixels(eq_xy,nn)
# InitCorr(dist_mat,Img,nn)

# print(">>>> Diffusion Map")
# eq_xy = two_d_eq(dm_xy)
# Img = Assign_features_to_pixels(eq_xy,nn,verbose=True)
# InitCorr(dist_mat,Img,nn)
#%%
#import pydiffmap as dm
#import ToolBox
#import progressbar
#results = []
#Paras = {'Epsilon': np.linspace(0.5,1.5,3),'Alpha':[1],'k':[16,32,64,128,256]}
#Para_table = ToolBox.grid_search_dict_to_df(Paras)
#p = progressbar.ProgressBar(max_value=len(Para_table))
#"""
#and since we want to unbias with respect to the non-uniform sampling density
# we set alpha = 1.0. The epsilon parameter controls the scale and needs to be
# adjusted to the data at hand. The k parameter controls the neighbour lists,
# a smaller k will increase performance but decrease accuracy.
#"""
#for ii in range(len(Para_table)):
#    each_para = Para_table.loc[ii]
#    metric = 'correlation'
#    eps = float(each_para['Epsilon'])
#    alpha = float(each_para['Alpha'])
#    k = int(each_para['k'])
#    p.update(ii)
#    print(eps,alpha)
#    try:
#        dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = metric,
#                                                           epsilon = eps,alpha=alpha,k = k)
#        dm_xy = dfmap.fit_transform(transposed_input)
#        eq_xy = two_d_eq(dm_xy)
#        img = Assign_features_to_pixels(eq_xy,nn,verbose=True)
#        results.append(InitCorr(dist_mat,img,nn))
#    except:
#        results.append('No Coverge')
#
#%% Plot test
#from matplotlib import pyplot as plt
#plt.scatter(tsne_xy[:,0],tsne_xy[:,1],s = 2)
#plt.title("tSNE")
#%% Save
# import pickle
# try:
#     with open('./Images/maps/Melanoma_KEGG_MDS.pickle','wb') as file:
#         pickle.dump([feature_names_list,dist_mat,Img],file)
#     print("File saved.")
# except IOError:
#     print("IO Error. Not saved.")
import pickle
try:
    with open('./Gauss_clusters/mapping_MDS.pickle','wb') as file:
        pickle.dump([feature_names_list,dist_mat,Img],file)
    print("File saved.")
except IOError:
    print("IO Error. Not saved.")
