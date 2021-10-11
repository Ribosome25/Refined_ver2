# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:56:33 2019

@author: Ruibo

"""
import os
import math
import pickle
from pickle import Pickler
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd

#from joblib import Parallel, delayed
#import multiprocessing
from myToolbox import ToolBox
from myToolbox.Stat import normalize_df
from imageio import imwrite as imsave
from scipy.stats import pearsonr
from scipy.spatial import distance
import sklearn.manifold as mnf
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
#%% Init corr func
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
    Dist_mat_Sq = distance.squareform(Dist_mat, checks=False)
    parent_pop2 = parent_pop          ## Added by Omid
    parent_pop2_vec = np.reshape(parent_pop2,(1,NN2)).tolist()[0]
    Indd2_vec = [parent_pop2_vec.index(i) for i in FF_FFN]
    PP_MM2_vec = PP_MM[np.ix_(Indd2_vec,Indd2_vec)]
    CORRr1, P_VAL = pearsonr(Dist_mat_Sq,
                             distance.squareform(PP_MM2_vec, checks=False))
    #CORRr1 = Cor2_arr[1,0]
    print("\nInitial: >>>> ",CORRr1)
    return CORRr1
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
        eq_xy[xx_idx,0] = ii * 1/len(xy)
        eq_xy[yy_idx,1] = ii * 1/len(xy)
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

    # xy = eq_xy
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
            print("\r>> Assign features to pixels: {} / {}".format(feature_assigned.sum(), Nn), end='')
    result_table = result_table.astype(int)

    img = np.full((nn,nn),'NaN').astype(object)
    for each_pixel in range(nn*nn):
        xx = result_table[each_pixel,0]
        yy = result_table[each_pixel,1]
        ft = 'F' + str(result_table[each_pixel,2])
        img[xx,yy] = ft
    return img.astype(object)
#%%
class Refined(object):
    def __init__(self, dim_reduction='MDS', distance_metric='correlation',
                 verbose=True, **kwarg):
        self.dist_m = distance_metric.lower()
        self.dim_r = dim_reduction.lower()
        self.verbose = verbose
        self.hw = None
        self._fitted = False
        self.args = kwarg

    @staticmethod
    def _transform_mapping_dict(init_map, feature_names_list):
        # init_map = self.mapping_obj_array.copy()
        int_map = np.char.strip(init_map.astype(str),'F').astype(int)
        coords = []
        for xx in range(int(np.max(int_map)+1)):  # TODO: changed here. Used to be stable.
            coord_in_arrays = np.where(int_map==xx)
            coord_in_list = [coord_in_arrays[0][0],coord_in_arrays[1][0]]
            coords.append(coord_in_list)
        pos_mat = np.array(coords)
        # self.mapping_dict = pd.DataFrame(pos_mat, index=self.feature_names_list, columns=['x','y']).to_dict('index')
        return pd.DataFrame(pos_mat, index=feature_names_list, columns=['x','y']).to_dict('index')


    def fit(self, original_input):
        """
        Calculate the initial correlations (distances) of features,
        Assign the original positions mapping.
        """
        assert isinstance(original_input, pd.DataFrame)
        feature_names_list = original_input.columns.tolist()
        nn = math.ceil(np.sqrt(len(feature_names_list))) # image dimension
        Nn = original_input.shape[1] # Feature amount total
        if 'corr' in self.dist_m:
            dist_mat = 1 - (original_input.corr()) # dist = 1-correlation
        else:
            raise ValueError("Initial distance metric. Now Corrs only.")
        original_input = normalize_df(original_input)
        transposed_input = original_input.T # dim-reduc, the fts as instances, data T as DF

        #%% Dim reduction:
        if 'mds' in self.dim_r:
            mds = mnf.MDS(dissimilarity='precomputed')
            xy = mds.fit_transform(dist_mat)
        elif 'isomap' in self.dim_r:
            isomap = mnf.Isomap(n_neighbors=25, n_components=2, eigen_solver='dense', path_method= 'D', n_jobs=3)
            xy = isomap.fit_transform(transposed_input)
        elif 'lle' in self.dim_r:
            lle = mnf.LocallyLinearEmbedding(n_neighbors=15)
            xy = lle.fit_transform(transposed_input)
        elif 'tsne' in self.dim_r:
            tsne = mnf.TSNE()
            xy = tsne.fit_transform(transposed_input)
        elif 'pca' in self.dim_r:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(transposed_input)
        elif 'dm' or 'diffusion' in self.dim_r:
            import pydiffmap as dm
            dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
                                                            alpha = 0.5, epsilon = 1, k = 32)
            xy = dfmap.fit_transform(transposed_input)
        if self.verbose:
            print(">>>> Dim-reduction done")
        #%% Assign pixels
        eq_xy = two_d_eq(xy)
        mapping = Assign_features_to_pixels(eq_xy,nn,verbose=self.verbose)
        try:
            InitCorr(dist_mat, mapping, nn)
        except ValueError:
            print("Refined fitting: Initial Corrs unavailable. ")
        #%%
        self.hw = nn
        self.feature_names_list = feature_names_list
        self.dist_mat = dist_mat
        self.mapping_obj_array = mapping
        # self._transform_mapping_dict()
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(), self.feature_names_list)
        self._fitted = True
        return self

    def save_mapping_for_hill_climbing(self, f_name='to_hill_climbing'):
        try:
            with open('./{}.pickle'.format(f_name),'wb') as file:
                pickle.dump([self.feature_names_list,
                    self.dist_mat, self.mapping_obj_array], file)
            print("File saved.")
        except IOError:
            print("IO Error. Not saved.")

    def load_from_hill_climbing(self, f_name='to_hill_climbing'):
        f_name = f_name.strip('.pickle')
        try:
            with open('{}.pickle'.format(f_name), 'rb') as file:
                feature_name_list, pos_mat, init_map = pickle.load(file)
            self._fitted = True
        except :
            print("Cannot read files from Hill climbing. ")
            return None
        self.feature_names_list = feature_name_list
        self.mapping_dict = pd.DataFrame(pos_mat, index=feature_name_list, columns=['x','y']).to_dict('index')
        self.mapping_obj_array = init_map
        return self

    def load_from_string_coords(self, string_f):
        if string_f.endswith('.csv'):
            tb = pd.read_csv(string_f, index_col=0)
        else:
            tb = pd.read_table(string_f, index_col=0)
        Nn = len(tb)
        nn = math.ceil(math.sqrt(Nn))
        self.feature_names_list = tb.index.tolist()
        xy = tb.iloc[:, :2].values
        eq_xy = two_d_eq(xy)
        mapping = Assign_features_to_pixels(eq_xy, nn, verbose=self.verbose)
        self.mapping_obj_array = mapping
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(),
                                                         self.feature_names_list)
        self._fitted = True
        self.hw = nn
        return self

    def generate_image(self, data_df, output_folder='Images', img_format='npy',
                       dtype=int, normalize_feature=True, zscore=False, zscore_cutoff=5,
                       random_map=False, white_noise=False):
        assert self._fitted
        assert isinstance(data_df, pd.DataFrame)
        gene_expression = data_df.copy()
        gene_expression.index = gene_expression.index.map(str) # In cases that index are ints.
        feature_name_list = self.feature_names_list.copy()
        # Check if features_names in REFINED is same to the given DF
        print("Features not found in DF: ", [x for x in feature_name_list if x not in gene_expression.columns])
        gene_expression = gene_expression.reindex(feature_name_list, axis="columns")
        mapping_dict = self.mapping_dict.copy()

        if random_map:
            import random
            random.seed(7)
            random.shuffle(feature_name_list)
            # mapping_dict = pd.DataFrame(pos_mat, index=feature_name_list, columns = ['x','y']).to_dict('index')
            df = pd.DataFrame.from_dict(self.mapping_dict, orient='index', columns=['x', 'y'])
            df.index = feature_name_list
            mapping_dict = df.to_dict('index')
        if white_noise:
            white_noise = np.random.gamma(4, 2, size=gene_expression.shape)
            gene_expression = pd.DataFrame(data=white_noise, index=gene_expression.index,
                                           columns=gene_expression.columns, dtype=int)

        if zscore:
            # temp = minmax_scale(scale(gene_expression), feature_range=(0,255), axis=0)
            temp = scale(gene_expression)
            temp = np.clip(temp, -zscore_cutoff, zscore_cutoff)
            # temp = minmax_scale(temp, feature_range=(0,255), axis=0)
            gene_expression = pd.DataFrame(temp,
                       index=gene_expression.index, columns=gene_expression.columns).fillna(0)
        elif normalize_feature:
            gene_expression = normalize_df(gene_expression, (0, 255)).fillna(0)
        # Normalize [0, 255]
        gene_expression = ToolBox.normalize_int_between(gene_expression, 0, 255)

        # Make folder
        try:
            os.mkdir('./'+output_folder)
        except FileExistsError:
            print("Folder already exists.")
        finally:
            os.chdir('./'+output_folder)

        # Save images
        n_img = len(gene_expression)
        debug_i = 1
        for idx, each_cell_line in gene_expression.iterrows():
            # each_cell_line is a Series.
            if self.verbose:
                print("\rGenerating images: {} / {}".format(debug_i, str(n_img)), end='')
            if ('np' in img_format) and (dtype == float):
                Img = np.zeros(self.mapping_obj_array.shape)
                # or start with 125?
                Img = np.full(self.mapping_obj_array.shape, np.nan)
            else:
                # imwrite requires unit8
                Img = np.zeros(self.mapping_obj_array.shape).astype(np.uint8)
                # or 125?
                Img = np.full(self.mapping_obj_array.shape, np.nan).astype(np.uint8)

            cell_line_name = each_cell_line.name
            # Exceptions:
            if cell_line_name == 'PE/CAPJ15':
                cell_line_name = 'PE'
            # save
            # for each_feature in each_cell_line.index:
            for each_feature in self.feature_names_list:
                xx = mapping_dict[each_feature]['x']
                yy = mapping_dict[each_feature]['y']
                val = each_cell_line[each_feature]
                Img[xx,yy] = val # note: here is defined as this, x is the rows, y is the cols.

            # OR use mean instead of 0?
            np.nan_to_num(Img, copy=False, nan=np.nanmean(Img))

            # Drug imgs does not accept underscore
            cell_line_name = cell_line_name.replace('_','')
            if 'VNLG124' == cell_line_name:
                cell_line_name = 'VNLG_124'

            if 'np' in img_format:# and dtype is float: （why have this dtype check?
                np.save('MDS_'+ str(cell_line_name), Img)
            else:
                imsave('MDS_'+cell_line_name+'.'+img_format, Img)

            debug_i += 1
        print("\n>>> Image generated.")
        return None

    def plot_mapping(self):
        import matplotlib.pyplot as plt
        assert self._fitted
        hw = self.mapping_obj_array.shape[0]
        mapp = self.mapping_dict
        for txt in self.feature_names_list:
            yy = mapp[txt]['y'] + 0.5
            xx = hw - mapp[txt]['x'] - 0.5
            plt.text(yy, xx, txt,
                    horizontalalignment='center', verticalalignment='center',
                    wrap=True)
        plt.grid()
        plt.xlim(0, hw)
        plt.ylim(0, hw)
        return None

    def reverse_mapping(self, array):
        pass
#%%
if __name__ == '__main__':
    with open('./data/GDSC_10fold.pickle', 'rb') as file:
        data = pickle.load(file)
        rna = data['rna_seq']
    gene = []
    # gene.append(pd.read_csv('./data/HH.csv', index_col=0, header=None))
    # gene.append(pd.read_csv('./data/NOTCH.csv', index_col=0, header=None))
    # gene.append(pd.read_csv('./data/TGFB.csv', index_col=0, header=None))
    gene.append(pd.read_csv('./data/Melanoma.csv', index_col=0, header=None))
    kegg = pd.concat(gene, axis=0).iloc[:, 0].unique()
    data = rna.reindex(columns=kegg).dropna(axis=1)

    process = Refined()
    process.fit(data)
    process.plot_mapping()
    process.save_mapping_for_hill_climbing()
    process.generate_image(data, 'Test', 'jpg')